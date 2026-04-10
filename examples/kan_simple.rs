/// KAN-NNUE training example.
///
/// Architecture: 768 -> ft(128) -> KAN(256->128) -> KAN(128->1) -> sigmoid
///
/// Uses B-spline KAN layers (trainable activation functions on edges)
/// instead of fixed activation functions like SCReLU or CReLU.
/// Validated in kanue to produce -22% loss and +2.1pp accuracy over baseline.
use bullet_lib::{
    game::inputs,
    nn::{
        kan::kan_layer,
        optimiser,
    },
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
const SPLINE_ORDER: usize = 3;
const SCALE: i32 = 400;

fn main() {
    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(optimiser::AdamW)
        .inputs(inputs::Chess768)
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(255),
            SavedFormat::id("l0b").round().quantise::<i16>(255),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs| {
            // Feature transformer: 768 -> FT_SIZE with SCReLU
            let l0 = builder.new_affine("l0", 768, FT_SIZE);
            let stm_ft = l0.forward(stm_inputs).screlu();
            let ntm_ft = l0.forward(ntm_inputs).screlu();
            let ft_out = stm_ft.concat(ntm_ft); // (2 * FT_SIZE, 1) batched

            // Clamp feature transformer output to [-1, 1] for B-spline grid range
            let clamped = ft_out.clip_pass_through_grad(-1.0, 1.0);

            // KAN layer 1: 2*FT_SIZE -> KAN_HIDDEN
            let kan1 = kan_layer(builder, "kan1", 2 * FT_SIZE, KAN_HIDDEN, GRID_SIZE, SPLINE_ORDER);
            let hidden = kan1.forward(clamped);

            // Clamp for second KAN layer
            let hidden_clamped = hidden.clip_pass_through_grad(-1.0, 1.0);

            // KAN layer 2: KAN_HIDDEN -> 1
            let kan2 = kan_layer(builder, "kan2", KAN_HIDDEN, 1, GRID_SIZE, SPLINE_ORDER);
            kan2.forward(hidden_clamped)
        });

    let schedule = TrainingSchedule {
        net_id: "kan-simple".to_string(),
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

    let data_loader = loader::DirectSequentialDataLoader::new(&["data/baseline.data"]);

    trainer.run(&schedule, &settings, &data_loader);
}
