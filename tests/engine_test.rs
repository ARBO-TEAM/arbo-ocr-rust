use std::path::PathBuf;
use std::time::Duration;

use arbo_ocr::{Config, Engine};

fn fake_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_fake_arboocr"))
}

fn engine_with_fake_bin() -> Engine {
    Engine::new(Config {
        bin_path: Some(fake_bin()),
        ..Default::default()
    })
    .expect("Engine::new with fake bin")
}

#[test]
fn recognize_parses_successful_json() {
    let engine = engine_with_fake_bin();
    let result = engine.recognize("/some/page.jpg").expect("recognize");

    assert_eq!(result.backend, "cpu");
    assert_eq!(result.image, "page.jpg");
    assert_eq!(result.elapsed_ms, 12.5);
    assert_eq!(result.lines.len(), 1);
    assert_eq!(result.lines[0].text, "hello");
    assert_eq!(result.lines[0].score, 0.9);
    assert_eq!(result.lines[0].det_score, 0.8);
    assert_eq!(result.lines[0].polygon.len(), 1);
    assert_eq!(result.lines[0].polygon[0].x, 1.0);
    assert_eq!(result.lines[0].polygon[0].y, 2.0);
}

#[test]
fn recognize_returns_error_on_non_zero_exit() {
    let engine = engine_with_fake_bin();

    let result = engine.recognize("FAIL");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.exit_code, Some(2));
    assert!(err.to_string().contains("exited with code"));
}

#[test]
fn recognize_returns_error_on_unparseable_output() {
    let engine = engine_with_fake_bin();
    let result = engine.recognize("GARBAGE");
    assert!(result.is_err());
}

#[test]
fn new_engine_errors_when_bin_path_missing() {
    let missing = std::env::temp_dir().join("no-such-arboocr-binary-xyz");
    let result = Engine::new(Config {
        bin_path: Some(missing),
        ..Default::default()
    });
    assert!(result.is_err());
}

#[test]
fn recognize_does_not_deadlock_on_large_stderr() {
    let engine = engine_with_fake_bin();

    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let result = engine.recognize("NOISY");
        let _ = tx.send(result);
    });

    match rx.recv_timeout(Duration::from_secs(10)) {
        Ok(Ok(result)) => {
            assert_eq!(result.backend, "cpu");
            assert_eq!(result.lines.len(), 1);
            assert_eq!(result.lines[0].text, "hello");
        }
        Ok(Err(e)) => panic!("recognize: {e}"),
        Err(_) => panic!("recognize deadlocked"),
    }
}
