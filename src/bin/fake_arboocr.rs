// Stand-in for arboocr_demo, used only by engine tests (via
// env!("CARGO_BIN_EXE_fake_arboocr")). Inspects the --image value (the real
// argv Engine::recognize builds) and dispatches on it — mirroring
// arbo-ocr-go's TESTFAIL/TESTGARBAGE/TESTNOISY sentinel-image-path pattern,
// since Config exposes no bare-flag test hook.
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // Prefetch mode takes no --image at all (the real binary fetches and
    // exits before it would open one), so it has to be dispatched before the
    // sentinel-image match below rather than as a branch inside it.
    if args.iter().any(|a| a == "--download-models") {
        println!("ok       det   /cache/PP-OCRv6_det.onnx");
        println!("skipped  cls   /cache/PP-OCRv6_cls.onnx");
        println!("ok       rec   /cache/PP-OCRv6_rec_small.onnx");
        println!("absent   dict  /cache/PP-OCRv6_rec_small_dict.txt");
        return;
    }

    let image = args
        .iter()
        .position(|a| a == "--image")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_default();

    match image.as_str() {
        "FAIL" => {
            eprintln!("simulated engine failure");
            std::process::exit(2);
        }
        "GARBAGE" => {
            println!("not json");
        }
        // The only shape that carries a "words" array. Every other branch
        // omits the key entirely, exactly as the real binary does when
        // --word-boxes is off — which is what LineResult's #[serde(default)]
        // has to survive.
        "WORDS" => {
            println!(
                "{{\"backend\":\"cpu\",\"image\":\"WORDS\",\"elapsedMs\":12.5,\"lines\":[{{\"text\":\"hi there\",\"score\":0.9,\"detScore\":0.8,\"polygon\":[{{\"x\":1.0,\"y\":2.0}}],\"words\":[{{\"text\":\"hi\",\"score\":0.95,\"polygon\":[{{\"x\":1.0,\"y\":2.0}}]}},{{\"text\":\"there\",\"score\":0.85,\"polygon\":[{{\"x\":3.0,\"y\":4.0}}]}}]}}]}}"
            );
        }
        "NOISY" => {
            // Past a pipe's OS buffer (~64KB), written before any stdout —
            // the real arboocr_demo does this via ONNXRuntime
            // schema-registration warnings.
            let stderr = std::io::stderr();
            let mut lock = stderr.lock();
            for _ in 0..20000 {
                let _ = lock.write_all(b"noise\n");
            }
            print_canned_json(&image);
        }
        _ => print_canned_json(&image),
    }
}

fn print_canned_json(image: &str) {
    let basename = std::path::Path::new(image)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    println!(
        "{{\"backend\":\"cpu\",\"image\":\"{basename}\",\"elapsedMs\":12.5,\"lines\":[{{\"text\":\"hello\",\"score\":0.9,\"detScore\":0.8,\"polygon\":[{{\"x\":1.0,\"y\":2.0}}]}}]}}"
    );
}
