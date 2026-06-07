from __future__ import annotations

from dataclasses import dataclass

from .features import FrameFeatures


@dataclass(slots=True)
class RuleAssessment:
    phase: str
    label: str
    score: float
    issues: list[str]
    suggestion: str


class SquatRuleEngine:
    def __init__(
        self,
        depth_knee_angle_threshold: float = 95.0,
        knee_valgus_ratio_factor: float = 0.85,
        torso_lean_threshold: float = 35.0,
    ) -> None:
        self.depth_knee_angle_threshold = depth_knee_angle_threshold
        self.knee_valgus_ratio_factor = knee_valgus_ratio_factor
        self.torso_lean_threshold = torso_lean_threshold

    def assess(self, features: FrameFeatures, phase: str) -> RuleAssessment:
        values = features.values
        avg_knee_angle = (values["left_knee_angle"] + values["right_knee_angle"]) / 2.0
        knee_gap_ratio = values["knee_gap_ratio"]
        hip_width_ratio = values["hip_width"] / max(values["shoulder_width"], 1e-6)
        torso_lean = abs(180.0 - values["torso_tilt_angle"])

        issues: list[str] = []
        score = 100.0

        if phase == "bottom" and avg_knee_angle > self.depth_knee_angle_threshold:
            issues.append("depth_insufficient")
            score -= 35.0

        if phase in {"descending", "bottom"} and knee_gap_ratio < hip_width_ratio * self.knee_valgus_ratio_factor:
            issues.append("knee_valgus")
            score -= 30.0

        if phase in {"descending", "bottom"} and torso_lean > self.torso_lean_threshold:
            issues.append("torso_lean")
            score -= 25.0

        if phase == "standing" and not issues:
            return RuleAssessment(
                phase=phase,
                label="ready",
                score=100.0,
                issues=[],
                suggestion="Start the next squat and keep your torso stable.",
            )

        if not issues:
            return RuleAssessment(
                phase=phase,
                label="good_form",
                score=max(score, 0.0),
                issues=[],
                suggestion="Good form. Keep knees aligned and move smoothly.",
            )

        priority = {
            "knee_valgus": ("knee_valgus", "Keep your knees aligned with your toes."),
            "torso_lean": ("torso_lean", "Lift your chest and keep your back straighter."),
            "depth_insufficient": ("depth_insufficient", "Go lower until your hips are closer to knee level."),
        }
        label, suggestion = priority[issues[0]]
        return RuleAssessment(
            phase=phase,
            label=label,
            score=max(score, 0.0),
            issues=issues,
            suggestion=suggestion,
        )

