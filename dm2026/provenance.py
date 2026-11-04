from hashlib import sha256

RESNET50_SHA256 = "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"


def file_sha256(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_weights(path):
    if file_sha256(path) != RESNET50_SHA256:
        raise ValueError(f"Expected official ResNet-50 IMAGENET1K_V2 weights: {path}")
