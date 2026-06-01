/// Phase E.1.h5 — Hybrid: KAN at output only, scaled to Akimbo v1.0 architecture.
///
/// Architecture:
///   (768x4hm -> 1024) FT with factoriser + 4 input buckets + horizontal king mirror
///   -> CReLU -> concat (2048 wide)
///   -> ReluKAN(2048, 1) G=5 k=3 with sample_range=grid_range=(0, 1)
///
/// Compared to phase_e0_smoke (Akimbo v1.0 CReLU baseline):
///   - 800 superbatches (full Phase E run, ~7.6h on A100 per the smoke result)
///   - Hybrid KAN topology: the v1.0 screlu(16) -> screlu(32) -> Linear(1) output stack
///     is replaced by a single ReLU-KAN layer 2048 -> 1
///   - No output buckets (matches kan_variant_e's pattern; engine LUT path is single-output)
///   - relu_kan_lut_save_format for kan_out (i8, Q_KAN=64, 64 samples,
///     [in][sample][out] layout — matching Phase D variant E LUT spec)
///   - save_rate=50 -> 16 checkpoints for SPRT-based checkpoint selection
use bullet_lib::{
    game::{
        formats::sfbinpack::{
            TrainingDataEntry,
            chess::{r#move::MoveType, piecetype::PieceType},
        },
        inputs::{ChessBucketsMirrored, get_num_buckets},
    },
    nn::{
        InitSettings, Shape,
        optimiser::{AdamW, AdamWParams},
        relu_kan::relu_kan_layer,
        relu_kan_lut::relu_kan_lut_save_format,
    },
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader},
};

const HL_SIZE: usize = 1024;
const GRID_SIZE: usize = 5;
const SUPPORT_WIDTH: usize = 3;
const SCALE: i32 = 400;
const QA: i16 = 255;
const Q_KAN: i16 = 64;
const NUM_LUT_SAMPLES: usize = 64;

fn main() {
    let initial_lr = 0.001;
    let final_lr = 0.001 * 0.3f32.powi(5);
    let superbatches = 800;
    let wdl_proportion = 0.75;

    #[rustfmt::skip]
    const BUCKET_LAYOUT: [usize; 32] = [
        0, 1, 2, 3,
        4, 4, 5, 5,
        6, 6, 6, 6,
        7, 7, 7, 7,
        8, 8, 8, 8,
        8, 8, 8, 8,
        9, 9, 9, 9,
        9, 9, 9, 9,
    ];

    const NUM_INPUT_BUCKETS: usize = get_num_buckets(&BUCKET_LAYOUT);

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .save_format(&[
            SavedFormat::id("l0w")
                .transform(|store, weights| {
                    let factoriser = store.get("l0f").values.repeat(NUM_INPUT_BUCKETS);
                    weights.into_iter().zip(factoriser).map(|(a, b)| a + b).collect()
                })
                .round()
                .quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            relu_kan_lut_save_format(
                "kan_out",
                2 * HL_SIZE,
                1,
                GRID_SIZE,
                SUPPORT_WIDTH,
                (0.0, 1.0),
                (0.0, 1.0),
                NUM_LUT_SAMPLES,
                Q_KAN,
            ),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs| {
            let l0f = builder.new_weights("l0f", Shape::new(HL_SIZE, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(NUM_INPUT_BUCKETS);

            let mut l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, HL_SIZE);
            l0.init_with_effective_input_size(32);
            l0.weights = l0.weights + expanded_factoriser;

            let stm_ft = l0.forward(stm_inputs).crelu();
            let ntm_ft = l0.forward(ntm_inputs).crelu();
            let ft_out = stm_ft.concat(ntm_ft);

            let kan_out = relu_kan_layer(
                builder,
                "kan_out",
                2 * HL_SIZE,
                1,
                GRID_SIZE,
                SUPPORT_WIDTH,
                (0.0, 1.0),
            );
            kan_out.forward(ft_out)
        });

    let stricter_clipping = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", stricter_clipping);
    trainer.optimiser.set_params_for_weight("l0f", stricter_clipping);

    let schedule = TrainingSchedule {
        net_id: "phase_e1_h5".to_string(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: wdl_proportion },
        lr_scheduler: lr::CosineDecayLR { initial_lr, final_lr, final_superbatch: superbatches },
        save_rate: 50,
    };

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 32 };

    let data_loader = {
        let file_path = "data/test80-2024-01-jan.binpack";
        let buffer_size_mb = 1024;
        let threads = 4;
        fn filter(entry: &TrainingDataEntry) -> bool {
            entry.ply >= 16
                && !entry.pos.is_checked(entry.pos.side_to_move())
                && entry.score.unsigned_abs() <= 3000
                && entry.mv.mtype() == MoveType::Normal
                && entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None
        }
        loader::SfBinpackLoader::new(file_path, buffer_size_mb, threads, filter)
    };

    trainer.run(&schedule, &settings, &data_loader);
}
