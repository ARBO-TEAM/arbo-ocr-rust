use serde::Deserialize;

/// One vertex of a [`LineResult`]'s polygon.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

/// One recognized text line. Polygon points are in the order arboOCR
/// reports them (clockwise from top-left-ish).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct LineResult {
    pub text: String,
    pub score: f64,
    #[serde(rename = "detScore")]
    pub det_score: f64,
    pub polygon: Vec<Point>,
}

/// A full-page OCR result — mirrors arboOCR's `PagePrediction`. An empty
/// `lines` vec is a normal, successful result (no text found), not an
/// error.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct PageResult {
    pub backend: String,
    pub image: String,
    #[serde(rename = "elapsedMs")]
    pub elapsed_ms: f64,
    pub lines: Vec<LineResult>,
}
