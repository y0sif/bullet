/// Phase A variant E — ReLU-KAN (arXiv 2406.02075).
///
/// Architecture: 768 -> ft(128) CReLU -> ReluKAN(256->128) -> ReluKAN(128->1)
/// Pure basis, no base path, per the paper.
///
/// The first layer receives CReLU output in [0, 1] — grid matches.
/// Output of layer 1 is clamped to [-1, 1] to match layer 2's grid range.
use bullet_lib::{
    game::{
        formats::sfbinpack::{
            TrainingDataEntry,
            chess::{r#move::MoveType, piecetype::PieceType},
        },
        inputs,
    },
    nn::{optimiser, relu_kan::relu_kan_layer},
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader},
};

const FT_SIZE: usize = 128;
const KAN_HIDDEN: usize = 128;
const GRID_SIZE: usize = 5;
const SUPPORT_WIDTH: usize = 3; // `k` in the ReLU-KAN paper
const SCALE: i32 = 400;
const QA: i16 = 255;

fn main() {
    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(optimiser::AdamW)
        .inputs(inputs::Chess768)
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs| {
            let l0 = builder.new_affine("l0", 768, FT_SIZE);
            let stm_ft = l0.forward(stm_inputs).crelu();
            let ntm_ft = l0.forward(ntm_inputs).crelu();
            let ft_out = stm_ft.concat(ntm_ft); // values in [0, 1]

            // ReLU-KAN layer 1: input range [0, 1] (CReLU output)
            let kan1 = relu_kan_layer(builder, "kan1", 2 * FT_SIZE, KAN_HIDDEN, GRID_SIZE, SUPPORT_WIDTH, (0.0, 1.0));
            let hidden = kan1.forward(ft_out);

            let hidden_clamped = hidden.clip_pass_through_grad(-1.0, 1.0);

            // ReLU-KAN layer 2: input range [-1, 1] (clipped hidden)
            let kan2 = relu_kan_layer(builder, "kan2", KAN_HIDDEN, 1, GRID_SIZE, SUPPORT_WIDTH, (-1.0, 1.0));
            kan2.forward(hidden_clamped)
        });

    let schedule = TrainingSchedule {
        net_id: "kan-variant-e".to_string(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 488,
            start_superbatch: 1,
            end_superbatch: 40,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.75 },
        lr_scheduler: lr::StepLR { start: 0.001, gamma: 0.1, step: 18 },
        save_rate: 10,
    };

    let settings = LocalSettings {
        threads: 4,
        test_set: None,
        output_directory: "checkpoints",
        batch_queue_size: 64,
    };

    let data_loader = {
        let file_path = "data/test77.binpack";
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
