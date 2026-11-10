#!/bin/bash
# Build and push the Drusilla image from the checked-out source tree.
# Run on a machine with Docker and `docker login` done (e.g. greif14).
#
#   docker/build_push.sh                 # main: tags <version> and latest
#                                        # any other branch: tags dev
#   TAG=dev docker/build_push.sh         # explicit tag; only the release tag
#                                        # (<pyproject version>) also moves latest
#   PUSH=0 docker/build_push.sh          # build only
#   IMAGE=myuser/drusilla docker/build_push.sh   # other Docker Hub repo
#   DOCKER="sudo docker" docker/build_push.sh
#
# Refuses to build from a dirty working tree so the image always matches a
# commit; set ALLOW_DIRTY=1 to override for a local test build.
#
# Users are told to pull :latest (README). Pipelines pin the release tag, the
# bare version (e.g. 0.3.0): Paludamentum compares it with the version of its
# drusilla submodule. :dev is the image of a branch under development and
# never moves :latest. The base image is the NGC TensorFlow build (several
# GB): the first build pulls it and the push takes a while.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DOCKER=${DOCKER:-docker}
IMAGE=${IMAGE:-gaiusaugustus/drusilla}
VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -1)
BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ -z "${TAG:-}" ]; then
    if [ "$BRANCH" = main ]; then TAG=$VERSION; else TAG=dev; fi
fi
PUSH=${PUSH:-1}
COMMIT=$(git rev-parse HEAD)

# The release tag also moves :latest; every other tag (dev, test builds) does not.
TAGS=("$TAG")
if [ "$TAG" = "$VERSION" ]; then
    TAGS+=(latest)
fi
TAG_ARGS=()
for t in "${TAGS[@]}"; do
    TAG_ARGS+=(-t "$IMAGE:$t")
done

# Everything docker/Dockerfile copies into the image (see .dockerignore) plus
# the build files themselves.
IMAGE_INPUTS="src model_cfg configs models pyproject.toml README.md LICENSE docker .dockerignore"
# shellcheck disable=SC2086  # IMAGE_INPUTS is a word list on purpose
if [ -n "$(git status --porcelain -- $IMAGE_INPUTS)" ] \
        && [ "${ALLOW_DIRTY:-0}" != 1 ]; then
    echo "Working tree has uncommitted changes in files that go into the image." >&2
    echo "Commit them first, or set ALLOW_DIRTY=1 for a throwaway test build." >&2
    exit 1
fi

echo "[$(date)] Building $IMAGE:{$(IFS=,; echo "${TAGS[*]}")} from $BRANCH@${COMMIT:0:7}"
$DOCKER build -f docker/Dockerfile \
    --build-arg DRUSILLA_GIT_COMMIT="$COMMIT" \
    "${TAG_ARGS[@]}" .

if [ "$PUSH" = 1 ]; then
    for t in "${TAGS[@]}"; do
        echo "[$(date)] Pushing $IMAGE:$t"
        $DOCKER push "$IMAGE:$t"
    done
fi

echo "[$(date)] Done."
echo ""
echo "Pull with:"
echo "  singularity build drusilla.sif docker://$IMAGE:$TAG"
