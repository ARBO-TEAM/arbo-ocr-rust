// Usage: cargo run --example smoke -- <models_dir> <image_path> [model_type]
use arbo_ocr::{Config, Engine};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: smoke <models_dir> <image_path> [model_type]");
        std::process::exit(1);
    }
    let models_dir = args[1].clone();
    let image_path = args[2].clone();
    let model_type = args.get(3).cloned().unwrap_or_else(|| "tiny".to_string());

    let engine = Engine::new(Config {
        models_dir: Some(models_dir),
        model_type: Some(model_type),
        ..Default::default()
    })?;

    let result = engine.recognize(&image_path)?;

    println!(
        "backend={} lines={} elapsedMs={:.1}",
        result.backend,
        result.lines.len(),
        result.elapsed_ms
    );
    for line in &result.lines {
        println!("  {:<40} score={:.3}", line.text, line.score);
    }
    Ok(())
}
