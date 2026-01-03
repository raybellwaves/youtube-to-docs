import regression_core

# Override the VIDEO_ID in the core module
regression_core.VIDEO_ID = "B0x2I_doX9o,Cu27fBy-kHQ"


def main():
    regression_core.clear_artifacts()

    # We force the model as per requirements, assuming it is available or we just run it.
    # The requirement says: "It should run ... -m bedrock-nova-2-lite-v1 -nys"
    model = "bedrock-nova-2-lite-v1"

    regression_core.run_regression(
        model=model,
        transcript_model=None,
        infographic_model=None,
        tts_model=None,
        no_youtube_summary=True,
    )

    regression_core.verify_output(
        model=model,
        transcript_model=None,
        infographic_model=None,
        tts_model=None,
        no_youtube_summary=True,
    )


if __name__ == "__main__":
    main()
