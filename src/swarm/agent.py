from __future__ import annotations


def build_prompt(
    current_value: int,
    neighbors_higher: int,
    neighbors_lower: int,
    neighbors_equal: int,
    min_value: int,
    max_value: int,
    p_i: float,
    grid_size: int,
    include_pi: bool = True,
) -> str:
    size_text = str(grid_size)
    range_text = f"Allowed range: [{min_value}, {max_value}] (width={max_value - min_value})\n"
    if include_pi:
        state_line = f"State: P_i (Acceptance)={p_i:.2f}, Current={current_value}. {range_text}"
        guidelines = (
            "Guidelines (Follow P_i for reaction magnitude):\n"
            "- Interpret P_i as susceptibility and (1 - P_i) as resistance.\n"
            "- If P_i is closer to 0, treat your current value as nearly fixed. "
            "You should almost never change it, even when neighbors disagree. "
            "Only change if all neighbors point in the same direction and the pressure feels overwhelming.\n"
            "- If P_i is in the middle, adapt but stay cautious: move when the neighborhood is clearly biased; stay if balanced.\n"
            "- If P_i is closer to 1, you are highly reactive and unstable. "
            "You should make bold, extreme changes, even overshooting, rather than inching toward consensus.\n"
            "- Decide direction (up, down, or stay) from neighbor counts, then choose a value consistent with that direction.\n"
            "- Step size should scale smoothly with P_i: low=tiny, mid=moderate, high=large (relative to the allowed range).\n"
            "- Constraints: Output must be within range. Value change must match rationale direction.\n\n"
        )
    else:
        state_line = f"State: Current={current_value}. {range_text}"
        guidelines = (
            "Guidelines:\n"
            "- Decide direction (up, down, or stay) from neighbor counts.\n"
            "- If the neighborhood is clearly biased, move in that direction; if balanced, stay or move slightly.\n"
            "- Use a tiny/moderate/large change based on how strong the pressure feels.\n"
            "- Constraints: Output must be within range. Value change must match rationale direction.\n\n"
        )
    prompt = (
        f"Role: Agent in a {size_text}x{size_text} torus grid. "
        "Goal: Reach global integer consensus using ONLY neighbor comparisons.\n"
        f"{state_line}"
        f"Neighbors vs Self: {neighbors_higher} higher, {neighbors_lower} lower, {neighbors_equal} equal.\n\n"
        f"{guidelines}"
        "Return ONLY a JSON object with keys in this order: rationale (1-2 sentences), value (integer)."
    )
    return prompt


def build_diffusion_prompt(
    current_value: int,
    neighbors_higher: int,
    neighbors_lower: int,
    neighbors_equal: int,
    p_i: float,
    min_value: int,
    max_value: int,
    grid_size: int,
    include_pi: bool = True,
) -> str:
    size_text = str(grid_size)
    if include_pi:
        state_line = f"State: P_i (Acceptance)={p_i:.2f}, Current={current_value}. Allowed range: [{min_value}, {max_value}]\n"
        guidelines = (
            "Guidelines (Follow P_i for reaction magnitude):\n"
            "- Interpret P_i as susceptibility and (1 - P_i) as resistance.\n"
            "- If P_i is closer to 0, treat your current value as nearly fixed. "
            "You should almost never change it, even when neighbors disagree. "
            "Only change if all neighbors point in the same direction and the pressure feels overwhelming.\n"
            "- If P_i is in the middle, adapt but stay cautious: move when the neighborhood is clearly biased; stay if balanced.\n"
            "- If P_i is closer to 1, you are highly reactive and unstable. "
            "You should make bold, extreme changes, even overshooting, rather than inching toward balance.\n"
            "- Decide direction (up, down, or stay) from neighbor counts, then choose a value consistent with that direction.\n"
            "- Step size should scale smoothly with P_i: low=tiny, mid=moderate, high=large (relative to the allowed range).\n"
            "- Constraints: Output must be within range. Value change must match rationale direction.\n\n"
        )
    else:
        state_line = f"State: Current={current_value}. Allowed range: [{min_value}, {max_value}]\n"
        guidelines = (
            "Guidelines:\n"
            "- Decide direction (up, down, or stay) from neighbor counts.\n"
            "- If the neighborhood is clearly biased, move in that direction; if balanced, stay or move slightly.\n"
            "- Use a tiny/moderate/large change based on how strong the pressure feels.\n"
            "- Constraints: Output must be within range. Value change must match rationale direction.\n\n"
        )
    prompt = (
        f"Role: Agent in a {size_text}x{size_text} grid with open boundaries. "
        "Goal: Create a smooth gradient using ONLY neighbor comparisons.\n"
        f"{state_line}"
        f"Neighbors vs Self: {neighbors_higher} higher, {neighbors_lower} lower, {neighbors_equal} equal.\n\n"
        f"{guidelines}"
        "Return ONLY a JSON object with keys in this order: rationale (1-2 sentences), value (integer)."
    )
    return prompt

