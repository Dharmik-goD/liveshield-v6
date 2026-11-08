# MOSIP Decode alignment

The 2026 problem statement calls for Desktop and Android Registration Clients to process face image streams from biometric devices, perform face liveness and PAD, and guide residents/operators/supervisors through active facial challenges when required.

LiveShield is organized around these seams:

1. **Capture adapter** — accepts image frames from the browser demo; replaceable with a certified biometric-device adapter.
2. **Human/face detection** — confirms a person/face is in the capture scene. Person detection is not identity verification.
3. **Quality gate** — rejects unusable captures before PAD.
4. **Passive PAD adapter** — model interface that can host an appropriately trained model.
5. **Challenge policy** — requests an unpredictable active action when PAD is unavailable or uncertain.
6. **Decision response** — returns structured guidance to the client.

Production integration requirements still outstanding: device SDK integration, Android packaging, secure client/server protocol, role-based Resident/Operator/Supervisor workflows, model signing/versioning, replay/injection threat controls, calibrated decision thresholds, robust sequence verification, and independent PAD testing.

The system should follow ISO/IEC 30107 terminology and testing principles. This prototype makes no certification claim.
