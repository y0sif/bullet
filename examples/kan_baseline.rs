/// Baseline NNUE training example (for comparison with KAN).
///
/// Architecture: 768 -> ft(128) -> SCReLU -> 256->128 -> SCReLU -> 128->1 -> sigmoid
///
/// Same dimensions as kan_simple.rs but using standard SCReLU activations
/// instead of KAN layers. Use this to measure the loss improvement from KAN.
use bullet_lib::{
    game::{
        formats::sfbinpack::{
            TrainingDataEntry,
            chess::{r#move::MoveType, piecetype::PieceType},
        },
        inputs,
    },
    nn::optimiser,
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader},
};

const FT_SIZE: usize = 128;
const HIDDEN: usize = 128;
const SCALE: i32 = 400;

fn main() {
    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(optimiser::AdamW)
        .inputs(inputs::Chess768)
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(255),
            SavedFormat::id("l0b").round().quantise::<i16>(255),
            SavedFormat::id("l1w").round().quantise::<i8>(64),
            SavedFormat::id("l1b"),
            SavedFormat::id("l2w").round().quantise::<i8>(64),
            SavedFormat::id("l2b"),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs| {
            // Feature transformer: 768 -> FT_SIZE with SCReLU (same as KAN)
            let l0 = builder.new_affine("l0", 768, FT_SIZE);
            let stm_ft = l0.forward(stm_inputs).screlu();
            let ntm_ft = l0.forward(ntm_inputs).screlu();
            let ft_out = stm_ft.concat(ntm_ft); // (2 * FT_SIZE, 1) batched

            // Hidden layer 1: 256 -> HIDDEN with SCReLU (replaces KAN layer 1)
            let l1 = builder.new_affine("l1", 2 * FT_SIZE, HIDDEN);
            let hidden = l1.forward(ft_out).screlu();

            // Output layer: HIDDEN -> 1 (replaces KAN layer 2)
            let l2 = builder.new_affine("l2", HIDDEN, 1);
            l2.forward(hidden)
        });

    let schedule = TrainingSchedule {
        net_id: "kan-baseline".to_string(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
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

    // Same test77 dataset as KAN experiment
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
