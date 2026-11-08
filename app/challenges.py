import secrets

CHALLENGES = (
    "Please blink naturally",
    "Please slowly turn your head to the left",
    "Please slowly turn your head to the right",
)


def choose_challenge() -> str:
    return secrets.choice(CHALLENGES)
