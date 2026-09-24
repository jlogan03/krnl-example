use std::time::Duration;

pub struct CaseSummary {
    name: String,
    inputs: usize,
    reference: f64,
    rows: Vec<Row>,
}

struct Row {
    variant: &'static str,
    execution: String,
    result: f64,
    compute: Duration,
    total: Option<Duration>,
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

    // Record existing measurements outside the timed regions.
    pub fn record(
        &mut self,
        variant: &'static str,
        execution: String,
        result: f64,
        compute: Duration,
        total: Option<Duration>,
    ) {
        self.rows.push(Row {
            variant,
            execution,
            result,
            compute,
            total,
        });
    }
}

pub fn print(cases: &[CaseSummary]) {
    println!(
        "\nBenchmark summary (times in ms, mean of {} runs)",
        super::TIMING_RUNS
    );
    println!("Compute: CPU reduction or GPU dispatch + wait. CPU N = N Rayon chunks.");
    println!("GPU total: mean upload + compute + one readback; excludes allocation and setup.");
    println!("Absolute errors are relative to each dataset's f64 reference.");
    for case in cases {
        println!(
            "\n{} ({} inputs; reference={:.12e})",
            case.name, case.inputs, case.reference
        );
        println!(
            "| Variant | Execution  | Result             | Abs error | Compute ms | GPU total ms |"
        );
        println!(
            "|---------|------------|--------------------|-----------|------------|--------------|"
        );
        for row in &case.rows {
            let total = row
                .total
                .map_or_else(|| "-".into(), |t| format!("{:.3}", t.as_secs_f64() * 1e3));
            println!(
                "| {:<7} | {:<10} | {:>18.12e} | {:>9.3e} | {:>10.3} | {:>12} |",
                row.variant,
                row.execution,
                row.result,
                (row.result - case.reference).abs(),
                row.compute.as_secs_f64() * 1e3,
                total,
            );
        }
    }
}
