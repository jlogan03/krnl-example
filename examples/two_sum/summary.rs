use krnl::anyhow::{Result, ensure};
use std::time::Duration;

pub struct CaseSummary {
    name: String,
    inputs: usize,
    reference: f64,
    rows: Vec<Row>,
}

struct Row {
    variant: &'static str,
    policy: &'static str,
    cpu: (f64, Duration),
    gpu: (f64, Duration),
    upload: Duration,
    download: Duration,
}

impl CaseSummary {
    pub fn new(name: &str, inputs: usize, reference: f64) -> Self {
        Self {
            name: name.into(),
            inputs,
            reference,
            rows: Vec::new(),
        }
    }

    pub fn record(
        &mut self,
        variant: &'static str,
        fast: bool,
        cpu: (f64, Duration),
        gpu: (f64, Duration),
        upload: Duration,
        download: Duration,
    ) -> Result<()> {
        let policy = if fast { "fast" } else { "strict" };
        ensure!(
            cpu.0.is_finite(),
            "CPU {variant} returned a nonfinite result"
        );
        ensure!(
            gpu.0.is_finite(),
            "GPU {variant}/{policy} returned a nonfinite result"
        );
        self.rows.push(Row {
            variant,
            policy,
            cpu,
            gpu,
            upload,
            download,
        });
        Ok(())
    }

    pub fn print(&self) {
        println!(
            "\n{} ({} inputs; reference={:.12e})",
            self.name, self.inputs, self.reference
        );
        println!(
            "| Variant         | GPU math | CPU abs error | GPU abs error |     cpu ms | gpu upload ms | gpu compute ms | gpu download ms |"
        );
        println!(
            "|-----------------|----------|---------------|---------------|------------|---------------|----------------|-----------------|"
        );
        let ms = |time: Duration| time.as_secs_f64() * 1e3;
        for row in &self.rows {
            println!(
                "| {:<15} | {:<8} | {:>13.3e} | {:>13.3e} | {:>10.3} | {:>13.3} | {:>14.3} | {:>15.3} |",
                row.variant,
                row.policy,
                (row.cpu.0 - self.reference).abs(),
                (row.gpu.0 - self.reference).abs(),
                ms(row.cpu.1),
                ms(row.upload),
                ms(row.gpu.1),
                ms(row.download),
            );
        }
    }
}
