# docker_build.sh
#!/bin/bash

# Function to print usage
print_usage() {
  echo "Usage: $0 [-i image] [-p architecture] [-b use_buildpack]"
  echo "  -i image           Docker image (e.g., <namespace>/<app-name>:<version>)"
  echo "  -p platform        Target platform architecture (linux/amd64, linux/arm64, ...)"
  echo "  -b                 Use buildpack instead of direct Docker build (optional)"
}

# Default values
ARCH="linux/amd64"
USE_BUILDPACK=false

# Parse command line arguments
while getopts 'i:p:b' flag; do
  case "${flag}" in
    i) IMAGE="${OPTARG}" ;;
    p) ARCH="${OPTARG}" ;;
    b) USE_BUILDPACK=true ;;
    *) print_usage
       exit 1 ;;
  esac
done

# Check if IMAGE is provided
if [ -z "$IMAGE" ]; then
  echo "Error: Image name is required"
  print_usage
  exit 1
fi

# derive tag for latest version
LATEST=$(echo $IMAGE | sed -e 's/\:.*$/:latest/g')

# Display arguments
echo "Platform: $ARCH"
echo "Image: $IMAGE"
echo "Using Buildpack: $USE_BUILDPACK"

if [ "$USE_BUILDPACK" = true ]; then
  echo "Building with buildpack..."
  pack build "$IMAGE" \
    --builder paketobuildpacks/builder:base \
    --path . \
    --env BP_DOCKERFILE=Dockerfile \
    --env BP_PLATFORM_API="$ARCH"
else
  echo "Building with Docker..."
  docker build --progress=plain \
    --platform "$ARCH" -t "$IMAGE" -t $LATEST .
fi