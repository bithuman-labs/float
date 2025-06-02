import face_alignment

face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D, flip_input=False, device="cpu"
)
