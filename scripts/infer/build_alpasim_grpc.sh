#!/usr/bin/env bash
# alpasim_grpc (AlpaSim's gRPC stubs) for environments its package does not install into (it requires python >= 3.11):
# compiles the protos of an AlpaSim checkout into TARGET with grpc_tools (pip install grpcio-tools==1.62.3, which
# matches protobuf 4.25), plus the package __init__ with its API version. Then add TARGET to PYTHONPATH.
#   bash scripts/infer/build_alpasim_grpc.sh /path/to/alpasim /path/to/target
set -euo pipefail

GRPC_DIR="${1:?AlpaSim checkout}/src/grpc"
TARGET="${2:?target directory}"
VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "${GRPC_DIR}/pyproject.toml")"
IFS=. read -r MAJOR MINOR PATCH <<< "${VERSION}"

mkdir -p "${TARGET}"
python -m grpc_tools.protoc -I "${GRPC_DIR}" --python_out="${TARGET}" --grpc_python_out="${TARGET}" \
  "${GRPC_DIR}"/alpasim_grpc/v0/*.proto
touch "${TARGET}/alpasim_grpc/v0/__init__.py"
cat > "${TARGET}/alpasim_grpc/__init__.py" <<EOF
from alpasim_grpc.v0.common_pb2 import VersionId

__version__ = (${MAJOR}, ${MINOR}, ${PATCH})
API_VERSION_MESSAGE = VersionId.APIVersion(major=${MAJOR}, minor=${MINOR}, patch=${PATCH})
EOF
echo "alpasim_grpc ${VERSION} -> ${TARGET}"
