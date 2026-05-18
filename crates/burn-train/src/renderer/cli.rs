use crate::metric::NumericEntry;
use crate::renderer::{
    EvaluationProgress, MetricState, MetricsRenderer, MetricsRendererEvaluation,
    MetricsRendererTraining, ProgressType, TrainingProgress,
};

/// A simple renderer for when the cli feature is not enabled.
pub struct CliMetricsRenderer;

#[allow(clippy::new_without_default)]
impl CliMetricsRenderer {
    /// Create a new instance.
    pub fn new() -> Self {
        Self {}
    }
}

impl MetricsRendererTraining for CliMetricsRenderer {
    fn update_train(&mut self, state: MetricState) {
        log_metric("train", state);
    }

    fn update_valid(&mut self, state: MetricState) {
        log_metric("val", state);
    }

    fn render_train(&mut self, item: TrainingProgress, _progress_indicators: Vec<ProgressType>) {
        println!("{item:?}");
    }

    fn render_valid(&mut self, item: TrainingProgress, _progress_indicators: Vec<ProgressType>) {
        println!("{item:?}");
    }
}

impl MetricsRendererEvaluation for CliMetricsRenderer {
    fn render_test(&mut self, item: EvaluationProgress, _progress_indicators: Vec<ProgressType>) {
        println!("{item:?}");
    }

    fn update_test(&mut self, _name: super::EvaluationName, _state: MetricState) {}
}

impl MetricsRenderer for CliMetricsRenderer {
    fn manual_close(&mut self) {
        // Nothing to do.
    }

    fn register_metric(&mut self, _definition: crate::metric::MetricDefinition) {}
}

fn log_metric(split: &str, state: MetricState) {
    match state {
        MetricState::Generic(entry) => {
            log::info!("[{split}] {}", entry.serialized_entry.formatted);
        }
        MetricState::Numeric(entry, numeric) => {
            let name = format!("{:?}", entry.metric_id);
            match numeric {
                NumericEntry::Value(value) => {
                    log::info!("[{split}] {name}: {value:.6}");
                }
                NumericEntry::Aggregated {
                    aggregated_value,
                    count,
                } => {
                    log::info!("[{split}] {name}: {aggregated_value:.6} (n={count})");
                }
            }
        }
    }
}
