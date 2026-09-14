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

    // --images-from is batch mode: one process over a newline-delimited list
    // file, one JSON array on stdout in list order. Each path is echoed back
    // as that page's line text, so the tests can assert positional matching
    // rather than assume it.
    //
    // The error paths are selected by sentinel *paths* rather than argv,
    // because the engine puts the list in a file — a sentinel flag would never
    // reach this process.
    if let Some(i) = args.iter().position(|a| a == "--images-from") {
        let raw = std::fs::read_to_string(args.get(i + 1).map(String::as_str).unwrap_or(""))
            .unwrap_or_default();
        // Sentinels stay in the list and get a page like any other path: they
        // are real entries to the engine, which counts them and matches by
        // position. Drop-last below is what creates the short array.
        let paths: Vec<&str> = raw
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .collect();

        // What a bad flag actually does: exit 1 with no JSON on stdout, which
        // must not be confused with the ordinary "a page came back empty"
        // exit 1 that still carries the array.
        if raw.lines().any(|l| l.trim().ends_with("--batch-usage-error")) {
            eprintln!("Option '--images-from' does not exist");
            std::process::exit(1);
        }

        // A short array: one page fewer than the caller asked for, which the
        // engine has to reject rather than match up by position.
        let drop_last = raw.lines().any(|l| l.trim().ends_with("--batch-short"));

        let mut pages: Vec<String> = paths
            .iter()
            .map(|p| {
                let basename = std::path::Path::new(p)
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                format!(
                    "{{\"backend\":\"cpu\",\"image\":\"{basename}\",\"elapsedMs\":12.5,\"lines\":[{{\"text\":\"{p}\",\"score\":0.9,\"detScore\":0.8,\"polygon\":[{{\"x\":1.0,\"y\":2.0}}]}}]}}"
                )
            })
            .collect();
        if drop_last {
            pages.pop();
        }
        println!("[{}]", pages.join(","));

        // A batch exits 1 when any image came back empty — an ordinary
        // outcome that still carries the JSON the caller asked for.
        if raw.lines().any(|l| l.trim().ends_with("--batch-exit1")) {
            std::process::exit(1);
        }
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
