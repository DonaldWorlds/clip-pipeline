"""ClientStyleProfile — the single source of truth for per-client render decisions.

The pipeline does NOT have taste of its own. Every rendering decision
(caption style, aspect, color grade, upscale on/off) is read from a
ClientStyleProfile derived from that client's reference clips at ingest
time. The pipeline is a delivery mechanism; the references are the brain.

Profile lifecycle:
  - Derived: rag/ingest.py runs `derive_from_references()` whenever new refs
    are added under references/<client>/, then writes references/<client>/profile.json.
  - Loaded: cut.py calls `load(client)` at startup; render + validate consume it.
  - User-editable: profile.json can be hand-tuned; ingest can refuse to
    overwrite with --no-overwrite-profile.

Floor defaults come from references/_global/style.md (parsed by
editorial/rules.py) when references are missing or insufficient.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from statistics import median

from .config import PROJECT_ROOT
from .editorial import rules as rules_mod


REFS_ROOT = PROJECT_ROOT / "references"
MIN_REFS_FOR_DERIVATION = 3


@dataclass
class CaptionStyle:
    mode: str = "karaoke_word"           # subtitles | lower_third | karaoke_word
    position: str = "lower_third"        # lower_third | center | upper_third
    font: str = "Impact"
    font_px_at_1080: int = 96
    primary_color: str = "&H00FFFFFF"
    highlight_color: str = "&H0000FF00"
    outline_color: str = "&H00000000"
    outline_width: int = 6
    y_frac: float = 0.78
    derivation_confidence: float = 0.0   # 0.0 = pure default, 1.0 = unanimous refs


@dataclass
class ColorGrade:
    brightness_target: float = 128.0     # 0-255 mean luminance
    saturation_target: float = 1.0       # 1.0 = neutral
    contrast_target: float = 1.0
    color_temp: str = "neutral"          # warm | neutral | cool


@dataclass
class ClientStyleProfile:
    """Per-client style profile. One profile per --client. Read once at
    cut.py startup; passed by reference to render, decisions, and validate."""
    client: str = "_default"
    derived_at: str = ""
    n_references: int = 0

    delivery_aspects: list[str] = field(default_factory=lambda: ["9:16"])
    typical_clip_length_s: float = 15.0

    caption: CaptionStyle = field(default_factory=CaptionStyle)
    color_grade: ColorGrade = field(default_factory=ColorGrade)

    use_upscaling: bool = False

    source_fingerprint_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "ClientStyleProfile":
        cap = raw.get("caption", {}) or {}
        cg = raw.get("color_grade", {}) or {}
        return cls(
            client=raw.get("client", "_default"),
            derived_at=raw.get("derived_at", ""),
            n_references=int(raw.get("n_references", 0)),
            delivery_aspects=list(raw.get("delivery_aspects") or ["9:16"]),
            typical_clip_length_s=float(raw.get("typical_clip_length_s", 15.0)),
            caption=CaptionStyle(**cap) if cap else CaptionStyle(),
            color_grade=ColorGrade(**cg) if cg else ColorGrade(),
            use_upscaling=bool(raw.get("use_upscaling", False)),
            source_fingerprint_ids=list(raw.get("source_fingerprint_ids") or []),
        )


def _default_from_rules() -> ClientStyleProfile:
    """Build a profile from references/_global/style.md only (no client refs)."""
    rules = rules_mod.load()
    cap = CaptionStyle(
        position=rules.caption_position,
        font_px_at_1080=rules_mod.caption_font_px(rules, 1080),
        highlight_color=_color_name_to_ass(rules.caption_highlight_color),
        y_frac=rules_mod.caption_y_fraction(rules, None),
    )
    return ClientStyleProfile(
        client="_default",
        caption=cap,
        typical_clip_length_s=(rules.clip_min_s + rules.clip_max_s) / 2,
    )


def _color_name_to_ass(name: str) -> str:
    """style.md uses plain color names; .ass uses &HBBGGRR. Minimal table."""
    table = {
        "yellow": "&H0000FFFF",
        "green":  "&H0000FF00",
        "white":  "&H00FFFFFF",
        "red":    "&H000000FF",
        "blue":   "&H00FF0000",
        "cyan":   "&H00FFFF00",
        "magenta":"&H00FF00FF",
    }
    return table.get(name.lower(), "&H0000FF00")


def profile_path_for(client: str) -> Path:
    return REFS_ROOT / client / "profile.json"


def load(client: str | None) -> ClientStyleProfile:
    """Load the profile for a client. Resolution order:
       1. references/<client>/profile.json (if present)
       2. derive from references/<client>/*.mp4 fingerprints (if ≥ MIN_REFS)
       3. defaults from references/_global/style.md (the floor)

    No --client argument → step 3 only. The pipeline still runs but with
    the global floor as style; no per-client tuning.
    """
    if not client:
        prof = _default_from_rules()
        print(f"[profile] no --client; using global defaults from style.md",
              file=sys.stderr)
        return prof

    pj = profile_path_for(client)
    if pj.is_file():
        try:
            prof = ClientStyleProfile.from_dict(json.loads(pj.read_text()))
            print(f"[profile] loaded {pj} ({prof.n_references} refs)",
                  file=sys.stderr)
            return prof
        except Exception as e:
            print(f"[profile] failed to read {pj}: {e}; falling back", file=sys.stderr)

    # Try derivation from raw fingerprints (rag.db) for this client.
    prof = derive_from_references(client)
    if prof is not None:
        save(prof)
        return prof

    # Floor.
    prof = _default_from_rules()
    prof.client = client
    print(f"[profile] no references yet for client={client}; using floor defaults",
          file=sys.stderr)
    return prof


def load_strict(client: str) -> ClientStyleProfile:
    """Production-mode loader: hard fail if the client has no profile and no
    derivable references. Use this when --client is supplied and the user
    expects the pipeline to refuse to render mediocre output for a client
    that hasn't been onboarded.

    Resolution order (no soft floor fallback):
      1. references/<client>/profile.json (if present)
      2. derive from references/<client>/*.mp4 fingerprints (≥ MIN_REFS)
      3. sys.exit with onboarding instructions
    """
    pj = profile_path_for(client)
    if pj.is_file():
        try:
            prof = ClientStyleProfile.from_dict(json.loads(pj.read_text()))
            print(f"[profile] loaded {pj} ({prof.n_references} refs)", file=sys.stderr)
            return prof
        except Exception as e:
            print(f"[profile] {pj} unreadable: {e}", file=sys.stderr)

    prof = derive_from_references(client)
    if prof is not None:
        save(prof)
        return prof

    refs_dir = REFS_ROOT / client
    sys.exit(
        f"no profile and < {MIN_REFS_FOR_DERIVATION} ingested references for "
        f"client={client!r}.\n"
        f"  Expected:    {pj}\n"
        f"  Refs dir:    {refs_dir} ({'exists' if refs_dir.is_dir() else 'missing'})\n"
        f"\n"
        f"To onboard this client, ingest reference clips first:\n"
        f"  python -m clip_pipeline.rag.ingest --client {client}\n"
        f"\n"
        f"Or, for a dev/demo run with floor defaults, use --quickstart instead "
        f"of --client."
    )


def save(profile: ClientStyleProfile) -> Path:
    pj = profile_path_for(profile.client)
    pj.parent.mkdir(parents=True, exist_ok=True)
    pj.write_text(json.dumps(profile.to_dict(), indent=2))
    print(f"[profile] wrote {pj}", file=sys.stderr)
    return pj


def derive_from_references(client: str) -> ClientStyleProfile | None:
    """Build a ClientStyleProfile from this client's ingested reference
    fingerprints. Returns None if < MIN_REFS_FOR_DERIVATION refs are
    available — caller falls back to style.md defaults.

    For Phase 5a Batch B this is the *minimum viable* derivation:
      - delivery_aspects from reference video dimensions
      - typical_clip_length_s from median duration
      - color_grade from VisualSignature (brightness, color_temp)
      - caption.y_frac, font_px from EditorialSignature
      - use_upscaling from a sharpness/resolution proxy

    Full derivation (caption_mode classification from Pegasus prose,
    frame_style detection) is iterated in subsequent passes.
    """
    try:
        from .rag.retrieve import list_fingerprints
    except ImportError:
        return None

    try:
        fps = list_fingerprints(client)
    except Exception as e:
        print(f"[profile] derive: list_fingerprints failed: {e}", file=sys.stderr)
        return None

    if not fps or len(fps) < MIN_REFS_FOR_DERIVATION:
        return None

    rules = rules_mod.load()
    floor = _default_from_rules()

    durations = [float(f.get("duration_s") or 0) for f in fps if f.get("duration_s")]
    typical_len = round(median(durations), 2) if durations else floor.typical_clip_length_s

    aspects = _aspects_from_fingerprints(fps)
    if not aspects:
        aspects = ["9:16"]

    visuals = [f.get("visual", {}) for f in fps if f.get("visual")]
    if visuals:
        from collections import Counter
        brightness_vals = [float(v.get("avg_brightness", 128.0)) for v in visuals]
        brightness = median(brightness_vals) if brightness_vals else floor.color_grade.brightness_target
        temps = [v.get("color_temp", "neutral") for v in visuals]
        temp = Counter(temps).most_common(1)[0][0] if temps else floor.color_grade.color_temp
    else:
        brightness = floor.color_grade.brightness_target
        temp = floor.color_grade.color_temp

    editorials = [f.get("editorial", {}) for f in fps if f.get("editorial")]
    if editorials:
        y_fracs = [float(e.get("caption_y_frac", floor.caption.y_frac)) for e in editorials]
        font_pxs_nonzero = [int(e.get("caption_font_px") or 0) for e in editorials]
        font_pxs_nonzero = [p for p in font_pxs_nonzero if p > 0]
        y_frac = round(median(y_fracs), 3) if y_fracs else floor.caption.y_frac
        # font_pxs_nonzero is usually empty because the fingerprinter doesn't
        # OCR raw refs (caption_font_px is hardcoded to 0). Fall through to
        # the floor in that case — never call median([]).
        font_px_from_refs = (
            int(median(font_pxs_nonzero)) if font_pxs_nonzero
            else floor.caption.font_px_at_1080
        )
        font_px = max(floor.caption.font_px_at_1080, font_px_from_refs)
    else:
        y_frac = floor.caption.y_frac
        font_px = floor.caption.font_px_at_1080

    # Upscaling: heuristic — if median visual.motion_score is low (clean
    # studio footage) AND we have ≥ 5 refs, lean toward upscaling on.
    motion_scores = [float(v.get("motion_score", 0.0)) for v in visuals]
    use_upscaling = (
        len(visuals) >= 5
        and motion_scores
        and median(motion_scores) < 8.0
    )

    profile = ClientStyleProfile(
        client=client,
        derived_at=_now_iso(),
        n_references=len(fps),
        delivery_aspects=aspects,
        typical_clip_length_s=typical_len,
        caption=CaptionStyle(
            position=rules.caption_position,
            font_px_at_1080=font_px,
            highlight_color=_color_name_to_ass(rules.caption_highlight_color),
            y_frac=y_frac,
            derivation_confidence=round(min(1.0, len(fps) / 10.0), 2),
        ),
        color_grade=ColorGrade(
            brightness_target=round(brightness, 1),
            saturation_target=1.0,   # TODO: derive from HSV mean S in a follow-up
            contrast_target=1.0,
            color_temp=temp,
        ),
        use_upscaling=use_upscaling,
        source_fingerprint_ids=[str(f.get("id", "?")) for f in fps],
    )
    return profile


def _aspects_from_fingerprints(fps: list[dict]) -> list[str]:
    """Aggregate aspect votes from reference dimensions. Each ref contributes
    its dominant_aspect to the set; we keep aspects appearing in ≥ 20% of refs."""
    if not fps:
        return []
    counts: dict[str, int] = {}
    for f in fps:
        ed = f.get("editorial") or {}
        a = ed.get("dominant_aspect")
        if a:
            counts[a] = counts.get(a, 0) + 1
    if not counts:
        return []
    threshold = max(1, len(fps) // 5)
    return [a for a, c in counts.items() if c >= threshold]


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
