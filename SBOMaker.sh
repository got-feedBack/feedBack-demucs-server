#!/bin/bash
# SBOM Update Script for SBOMaker
# Generates Software Bill of Materials for all components
# Usage: ./SBOMaker.sh [--webui] [--mobile] [--docker] [--all] [--no-fail]

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT="$(pwd)"
if [[ ! -f "$PROJECT_ROOT/.sbom-config.json" ]]; then
    # Check directory above the script's location
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PARENT_DIR="$(dirname "$SCRIPT_DIR")"
    if [[ -f "$PARENT_DIR/.sbom-config.json" ]]; then
        PROJECT_ROOT="$PARENT_DIR"
    fi
fi

CONFIG_FILE="$PROJECT_ROOT/.sbom-config.json"

# Default values
SBOM_DIR="$PROJECT_ROOT/sbom"
TIMESTAMP=$(date +%Y-%m-%d)
TIMESTAMP_FULL=$(date +%Y-%m-%d-%H%M)
CYCLONEDX_SCHEMA_VERSION="1.6"
DOCKER_IMAGE=""

# Component paths (discovered)
WEBUI_PATH=""
MOBILE_PATH=""
ANDROID_PATH=""
IOS_PATH=""
DOCKER_PATH=""
GENERIC_APPS="" # Format: "label1:path1 label2:path2"

# Component exclusions (defaults)
WEBUI_EXCLUDES='["**/node_modules/**", "**/.git/**", "**/build/**", "**/.next/**"]'
MOBILE_EXCLUDES='["**/node_modules/**", "**/.git/**"]'
IOS_EXCLUDES='["**/.git/**"]'

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Flags for component selection
GENERATE_WEBUI=false
GENERATE_MOBILE=false
GENERATE_DOCKER=false
GENERATE_ALL=true  # Default to all
NO_FAIL=false

# Dependency tracking
CRITICAL_DEPS=0
RECOMMENDED_DEPS=0
AVAILABLE_DEPS=0
DEPS_DATA_JSON="{}"
DEPS_DATA_MD=""

# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

load_config() {
    if [[ -f "$CONFIG_FILE" ]]; then
        log "Loading configuration from $CONFIG_FILE"
        
        # Load SBOM DIR
        local config_sbom_dir
        config_sbom_dir=$(jq -r '.sbom_dir // empty' "$CONFIG_FILE")
        if [[ -n "$config_sbom_dir" ]]; then
            if [[ "$config_sbom_dir" == *".."* ]] || [[ "$config_sbom_dir" == /* ]]; then
                log_error "Invalid sbom_dir in config: $config_sbom_dir (Path traversal detected)"
                exit 1
            fi
            SBOM_DIR="$PROJECT_ROOT/$config_sbom_dir"
        fi

        # Load Project Name for report (optional, can be added to report)
        # PROJECT_NAME=$(jq -r '.project_name // "Unknown Project"' "$CONFIG_FILE")

        # Load Web UI config
        if [[ $(jq '.webui // empty' "$CONFIG_FILE") != "" ]]; then
            [[ $(jq '.webui.path // empty' "$CONFIG_FILE") != "" ]] && WEBUI_PATH=$(jq -r '.webui.path' "$CONFIG_FILE")
            [[ $(jq '.webui.excludes // empty' "$CONFIG_FILE") != "" ]] && WEBUI_EXCLUDES=$(jq -c '.webui.excludes' "$CONFIG_FILE")
        fi

        # Load Mobile config
        if [[ $(jq '.mobile // empty' "$CONFIG_FILE") != "" ]]; then
            [[ $(jq '.mobile.path // empty' "$CONFIG_FILE") != "" ]] && MOBILE_PATH=$(jq -r '.mobile.path' "$CONFIG_FILE")
            [[ $(jq '.mobile.ios_path // empty' "$CONFIG_FILE") != "" ]] && IOS_PATH=$(jq -r '.mobile.ios_path' "$CONFIG_FILE")
            [[ $(jq '.mobile.excludes // empty' "$CONFIG_FILE") != "" ]] && MOBILE_EXCLUDES=$(jq -c '.mobile.excludes' "$CONFIG_FILE")
        fi

        # Load Docker config
        if [[ $(jq '.docker // empty' "$CONFIG_FILE") != "" ]]; then
            [[ $(jq '.docker.image // empty' "$CONFIG_FILE") != "" ]] && DOCKER_IMAGE=$(jq -r '.docker.image' "$CONFIG_FILE")
        fi

        # Load Generic Apps config
        if [[ $(jq '.generic // empty' "$CONFIG_FILE") != "" ]]; then
            GENERIC_APPS=$(jq -r '.generic | to_entries | .[] | "\(.key):\(.value)"' "$CONFIG_FILE" | tr '\n' ' ')
        fi
        
        log_success "Configuration loaded successfully"
    else
        log "No configuration file found at $CONFIG_FILE. Using defaults."
    fi
}

# =============================================================================
# LOGGING FUNCTIONS
# =============================================================================

init_log() {
    mkdir -p "$SBOM_DIR"
    LOG_FILE="$SBOM_DIR/update-log-$TIMESTAMP_FULL.txt"
    echo "=== SBOM Update Log ===" > "$LOG_FILE"
    echo "Date: $(date)" >> "$LOG_FILE"
    echo "Project: $PROJECT_ROOT" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
}

log() {
    local message="[$(date +%H:%M:%S)] $1"
    [[ -n "${LOG_FILE:-}" ]] && echo "$message" >> "${LOG_FILE:-}" || true
    echo -e "${BLUE}ℹ️  $1${NC}" >&2
}

log_success() {
    local message="[$(date +%H:%M:%S)] ✓ $1"
    [[ -n "${LOG_FILE:-}" ]] && echo "$message" >> "${LOG_FILE:-}" || true
    echo -e "${GREEN}✓ $1${NC}" >&2
}

log_warning() {
    local message="[$(date +%H:%M:%S)] ⚠ $1"
    [[ -n "${LOG_FILE:-}" ]] && echo "$message" >> "${LOG_FILE:-}" || true
    echo -e "${YELLOW}⚠ $1${NC}" >&2
}

log_error() {
    local message="[$(date +%H:%M:%S)] ✗ $1"
    [[ -n "${LOG_FILE:-}" ]] && echo "$message" >> "${LOG_FILE:-}" || true
    echo -e "${RED}✗ $1${NC}" >&2
}

log_section() {
    if [[ -n "${LOG_FILE:-}" ]]; then
        echo "" >> "${LOG_FILE:-}"
        echo "=== $1 ===" >> "${LOG_FILE:-}"
    fi
    echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}" >&2
    echo -e "${CYAN}  $1${NC}" >&2
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}" >&2
}

# =============================================================================
# TOOL CHECK AND INSTALL FUNCTIONS
# =============================================================================

check_tool() {
    local tool="$1"
    if command -v "$tool" &> /dev/null; then
        log "Found $tool: $(command -v "$tool")"
        return 0
    else
        return 1
    fi
}


install_syft() {
    log_warning "syft not found. Installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &> /dev/null; then
            brew install syft
        else
            log_error "Homebrew not found. Please install syft manually."
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        local tmp_install_syft=$(mktemp)
        curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh -o "$tmp_install_syft"
        sh "$tmp_install_syft" -s -- -b /usr/local/bin
        rm -f "$tmp_install_syft"
    else
        log_error "Unsupported OS. Please install syft manually."
        exit 1
    fi
}

install_grype() {
    log_warning "grype not found. Installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &> /dev/null; then
            brew install grype
        else
            log_error "Homebrew not found. Please install grype manually."
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        local tmp_install_grype=$(mktemp)
        curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh -o "$tmp_install_grype"
        sh "$tmp_install_grype" -s -- -b /usr/local/bin
        rm -f "$tmp_install_grype"
    else
        log_error "Unsupported OS. Please install grype manually."
        exit 1
    fi
}

check_dependencies() {
    log_section "Checking Dependencies"
    local missing_tools=()
    if ! check_tool "syft"; then missing_tools+=("syft"); fi
    if ! check_tool "jq"; then missing_tools+=("jq"); fi
    if ! check_tool "grype"; then missing_tools+=("grype"); fi

    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_warning "Missing tools: ${missing_tools[*]}"
        for tool in "${missing_tools[@]}"; do
            case "$tool" in
                syft) install_syft ;;
                grype) install_grype ;;
                jq)
                    log_warning "jq not found. Please install it manually."
                    log_error "  macOS: brew install jq"
                    log_error "  Linux: sudo apt-get install jq"
                    exit 1
                    ;;
            esac
        done
    fi
    log_success "All dependencies satisfied"
}

# =============================================================================
# DEPENDENCY CHECKING FUNCTIONS
# =============================================================================

check_outdated_deps() {
    local dir="$1"
    local component="$2"
    log_section "Checking Outdated Dependencies: $component"

    if [[ ! -d "$dir" ]]; then
        log_warning "Directory $dir not found. Skipping dependency check."
        return 1
    fi

    log "Running npm outdated in $dir..."
    local tmp_out
    tmp_out=$(mktemp)
    
    if ! (cd "$dir" && npm outdated --json > "$tmp_out" 2>/dev/null); then
        :
    fi

    if [[ ! -s "$tmp_out" ]] || [[ "$(cat "$tmp_out")" == "{}" ]]; then
        log_success "All dependencies for $component are up to date!"
        rm -f "$tmp_out"
        return 0
    fi

    log_warning "Found outdated dependencies for $component. Categorizing..."

    local comp_deps_json
    comp_deps_json=$(jq -c 'to_entries | map({
        package: .key,
        current: .value.current,
        latest: .value.latest,
        type: (if .value.current == .value.latest then "FIXED" else "UPDATE" end),
        severity: (
            if .value.current == null then "UNKNOWN"
            elif (.value.latest | split(".") | .[0] | tonumber) > (.value.current | split(".") | .[0] | tonumber) then "MAJOR"
            elif (.value.latest | split(".") | .[1] | tonumber) > (.value.current | split(".") | .[1] | tonumber) then "MINOR"
            else "PATCH" end
        )
    })' "$tmp_out")

    local maj_count=$(echo "$comp_deps_json" | jq '[.[] | select(.severity == "MAJOR")] | length')
    local min_count=$(echo "$comp_deps_json" | jq '[.[] | select(.severity == "MINOR")] | length')
    local pat_count=$(echo "$comp_deps_json" | jq '[.[] | select(.severity == "PATCH")] | length')

    CRITICAL_DEPS=$((CRITICAL_DEPS + maj_count))
    RECOMMENDED_DEPS=$((RECOMMENDED_DEPS + min_count + pat_count))
    AVAILABLE_DEPS=$((AVAILABLE_DEPS + maj_count + min_count + pat_count))

    local comp_md
    comp_md=$(echo "$comp_deps_json" | jq -r '.[] | "| \(.package) | \(.current // "N/A") | \(.latest) | \(.severity) |"')
    DEPS_DATA_MD="${DEPS_DATA_MD}\n### $component\n| Package | Current | Latest | Type |\n|---------|---------|--------|------|\n${comp_md}\n"

    if [[ -z "$DEPS_DATA_JSON" ]]; then
        DEPS_DATA_JSON=$(jq -n --arg comp "$component" --argjson deps "$comp_deps_json" '{($comp): $deps}')
    else
        DEPS_DATA_JSON=$(echo "$DEPS_DATA_JSON" | jq --arg comp "$component" --argjson deps "$comp_deps_json" '. + {($comp): $deps}')
    fi

    log_success "Found $(echo "$comp_deps_json" | jq 'length') outdated packages for $component"
    rm -f "$tmp_out"
    return 0
}

# =============================================================================
# SBOM GENERATION FUNCTIONS
# =============================================================================

generate_webui_sbom() {
    log_section "Generating Web UI SBOM"
    mkdir -p "$SBOM_DIR/webui"
    local output_file="$SBOM_DIR/webui/sbom-webui-$TIMESTAMP.json"
    local output_spdx="$SBOM_DIR/webui/sbom-webui-$TIMESTAMP.spdx.json"

    if [[ ! -f "$WEBUI_PATH/package.json" ]]; then
        log_error "package.json not found in webui path: $WEBUI_PATH"
        return 1
    fi

    log "Scanning $WEBUI_PATH for Web UI dependencies..."
    
    # Convert JSON array of excludes to syft arguments
    local exclude_args=()
    while read -r exclude; do
        exclude_args+=(--exclude "$exclude")
    done < <(echo "$WEBUI_EXCLUDES" | jq -r '.[]')

    if syft dir:"$WEBUI_PATH" \
        --output cyclonedx-json="$output_file" \
        "${exclude_args[@]}"; then
        log_success "Web UI SBOM (CycloneDX) generated: $output_file"
    else
        log_error "Failed to generate Web UI SBOM"
        return 1
    fi

    if syft dir:"$WEBUI_PATH" \
        --output spdx-json="$output_spdx" \
        "${exclude_args[@]}"; then
        log_success "Web UI SBOM (SPDX) generated: $output_spdx"
    else
        log_warning "Failed to generate Web UI SBOM in SPDX format"
    fi

    update_symlink "$SBOM_DIR/sbom.json" "$output_file"
    check_outdated_deps "$WEBUI_PATH" "Web UI"
    echo "$output_file"
}


generate_mobile_sbom() {
    log_section "Generating Mobile App SBOM"
    mkdir -p "$SBOM_DIR/mobile"
    local output_file="$SBOM_DIR/mobile/sbom-mobile-$TIMESTAMP.json"
    local output_spdx="$SBOM_DIR/mobile/sbom-mobile-$TIMESTAMP.spdx.json"

    if [[ ! -d "$MOBILE_PATH" ]]; then
        log_warning "Mobile directory not found: $MOBILE_PATH"
        return 1
    fi

    if [[ ! -f "$MOBILE_PATH/package.json" ]]; then
        log_error "package.json not found in mobile directory: $MOBILE_PATH"
        return 1
    fi

    log "Scanning $MOBILE_PATH for mobile dependencies..."
    
    # Convert JSON array of excludes to syft arguments
    local exclude_args=()
    while read -r exclude; do
        exclude_args+=(--exclude "$exclude")
    done < <(echo "$MOBILE_EXCLUDES" | jq -r '.[]')

    if syft dir:"$MOBILE_PATH" \
        --output cyclonedx-json="$output_file" \
        "${exclude_args[@]}"; then
        log_success "Mobile App SBOM (CycloneDX) generated: $output_file"
    else
        log_error "Failed to generate Mobile App SBOM"
        return 1
    fi

    if syft dir:"$MOBILE_PATH" \
        --output spdx-json="$output_spdx" \
        "${exclude_args[@]}"; then
        log_success "Mobile App SBOM (SPDX) generated: $output_spdx"
    else
        log_warning "Failed to generate Mobile App SBOM in SPDX format"
    fi

    if [[ -d "$IOS_PATH" ]]; then
        log "iOS directory found, generating iOS SBOM..."
        mkdir -p "$SBOM_DIR/ios"
        local ios_output="$SBOM_DIR/ios/sbom-ios-$TIMESTAMP.json"
        
        local ios_exclude_args=()
        while read -r exclude; do
            ios_exclude_args+=(--exclude "$exclude")
        done < <(echo "$IOS_EXCLUDES" | jq -r '.[]')

        if syft dir:"$IOS_PATH" \
            --output cyclonedx-json="$ios_output" \
            "${ios_exclude_args[@]}"; then
            log_success "iOS SBOM generated: $ios_output"
            update_symlink "$SBOM_DIR/sbom-ios.json" "$ios_output"
        fi
    fi

    update_symlink "$SBOM_DIR/sbom-mobile.json" "$output_file"
    check_outdated_deps "$MOBILE_PATH" "Mobile App"
    echo "$output_file"
}


generate_docker_sbom() {
    log_section "Generating Docker Image SBOM"
    mkdir -p "$SBOM_DIR/docker"
    local output_file="$SBOM_DIR/docker/sbom-docker-$TIMESTAMP.json"

    if ! docker image inspect "$DOCKER_IMAGE" &> /dev/null; then
        log_warning "Docker image not found: $DOCKER_IMAGE"
        if [[ -f "$PROJECT_ROOT/docker-compose.yaml" ]] || [[ -f "$PROJECT_ROOT/docker-compose-dev.yaml" ]]; then
            local compose_file="$PROJECT_ROOT/docker-compose.yaml"
            [[ -f "$PROJECT_ROOT/docker-compose-dev.yaml" ]] && compose_file="$PROJECT_ROOT/docker-compose-dev.yaml"
            local image_from_compose=$(grep -A5 "^  app:" "$compose_file" 2>/dev/null | grep "image:" | head -1 | awk '{print $2}')
            if [[ -n "$image_from_compose" ]]; then
                DOCKER_IMAGE="$image_from_compose"
                log "Found image in docker-compose: $DOCKER_IMAGE"
            fi
        fi
        if [[ -z "$DOCKER_IMAGE" ]]; then
            log_error "No Docker image specified and no docker-compose file found. Please use --docker-image <image_name>."
            return 1
        elif ! docker image inspect "$DOCKER_IMAGE" &> /dev/null; then
            log_error "Docker image not available. Please build the image first."
            return 1
        fi
    fi

    log "Scanning Docker image: $DOCKER_IMAGE"
    if syft "$DOCKER_IMAGE" \
        --output cyclonedx-json="$output_file"; then
        log_success "Docker SBOM (CycloneDX) generated: $output_file"
    else
        log_error "Failed to generate Docker SBOM"
        return 1
    fi

    update_symlink "$SBOM_DIR/sbom-docker.json" "$output_file"
    echo "$output_file"
}

generate_generic_sbom() {
    local label="$1"
    local path="$2"
    log_section "Generating Generic SBOM: $label"
    
    mkdir -p "$SBOM_DIR/generic/$label"
    local output_file="$SBOM_DIR/generic/$label/sbom-$label-$TIMESTAMP.json"
    
    log "Scanning $path for $label dependencies..."
    if syft dir:"$path" --output cyclonedx-json="$output_file"; then
        log_success "Generic SBOM ($label) generated: $output_file"
    else
        log_error "Failed to generate Generic SBOM for $label"
        return 1
    fi
    
    update_symlink "$SBOM_DIR/sbom-generic-$label.json" "$output_file"
    echo "$output_file"
}

discover_components() {
    log_section "Discovering Components"
    
    # Webapp: package.json
    if [[ -z "$WEBUI_PATH" ]]; then
        WEBUI_PATH=$(find "$PROJECT_ROOT" -maxdepth 3 -name "package.json" -exec dirname {} \; | head -n 1)
        [[ -n "$WEBUI_PATH" ]] && log_success "Webapp discovered at: $WEBUI_PATH" || log_warning "Webapp marker (package.json) not found"
    else
        log "Using configured Webapp path: $WEBUI_PATH"
    fi

    # Mobile: package.json with android/ or ios/ folders
    if [[ -z "$MOBILE_PATH" ]]; then
        if [[ -d "$PROJECT_ROOT/mobile" && -f "$PROJECT_ROOT/mobile/package.json" ]]; then
            MOBILE_PATH="$PROJECT_ROOT/mobile"
            log_success "Mobile App root discovered at: $MOBILE_PATH"
        else
            MOBILE_PATH=$(find "$PROJECT_ROOT" -maxdepth 3 -not -path "*/node_modules/*" -not -path "*/.next/*" -name "package.json" -exec sh -c 'dir=$(dirname "$1"); cd "$dir" && { [ -d "android" ] || [ -d "ios" ]; }' sh {} \; -exec dirname {} \; | head -n 1)
            [[ -n "$MOBILE_PATH" ]] && log_success "Mobile App root discovered at: $MOBILE_PATH" || log_warning "Mobile App root not found"
        fi
    else
        log "Using configured Mobile path: $MOBILE_PATH"
    fi

    # Android: AndroidManifest.xml or build.gradle
    if [[ -z "$ANDROID_PATH" ]]; then
        if [[ -n "$MOBILE_PATH" ]]; then
            ANDROID_PATH="$MOBILE_PATH/android"
            [[ -d "$ANDROID_PATH" ]] && log_success "Android discovered at: $ANDROID_PATH" || log_warning "Android folder not found in $MOBILE_PATH"
        else
            ANDROID_PATH=$(find "$PROJECT_ROOT" -maxdepth 3 \( -name "AndroidManifest.xml" -o -name "build.gradle" \) -exec dirname {} \; | head -n 1)
            [[ -n "$ANDROID_PATH" ]] && log_success "Android discovered at: $ANDROID_PATH" || log_warning "Android marker not found"
        fi
    else
        log "Using configured Android path: $ANDROID_PATH"
    fi

    # iOS: Podfile or .xcodeproj
    if [[ -z "$IOS_PATH" ]]; then
        if [[ -n "$MOBILE_PATH" ]]; then
            IOS_PATH="$MOBILE_PATH/ios"
            [[ -d "$IOS_PATH" ]] && log_success "iOS discovered at: $IOS_PATH" || log_warning "iOS folder not found in $MOBILE_PATH"
        else
            IOS_PATH=$(find "$PROJECT_ROOT" -maxdepth 3 \( -name "Podfile" -o -name "*.xcodeproj" \) -exec dirname {} \; | head -n 1)
            [[ -n "$IOS_PATH" ]] && log_success "iOS discovered at: $IOS_PATH" || log_warning "iOS marker not found"
        fi
    else
        log "Using configured iOS path: $IOS_PATH"
    fi

    # Docker: docker-compose.yaml or Dockerfile
    if [[ -z "$DOCKER_PATH" ]]; then
        DOCKER_PATH=$(find "$PROJECT_ROOT" -maxdepth 3 \( -name "docker-compose.yaml" -o -name "Dockerfile" \) -exec dirname {} \; | head -n 1)
        [[ -n "$DOCKER_PATH" ]] && log_success "Docker discovered at: $DOCKER_PATH" || log_warning "Docker marker not found"
    else
        log "Using configured Docker path: $DOCKER_PATH"
    fi

    # Generic Apps Discovery
    log "Discovering generic applications..."
    local markers=(
        "python:requirements.txt"
        "python:pyproject.toml"
        "python:setup.py"
        "go:go.mod"
        "rust:Cargo.toml"
        "java-maven:pom.xml"
        "java-gradle:build.gradle"
    )

    for marker in "${markers[@]}"; do
        local label="${marker%%:*}"
        local pattern="${marker#*:}"
        
        local found_path=$(find "$PROJECT_ROOT" -maxdepth 4 \
            -not -path "*/node_modules/*" \
            -not -path "*/.git/*" \
            -not -path "*/.next/*" \
            -name "$pattern" -exec dirname {} \; | head -n 1)
        
        if [[ -n "$found_path" ]]; then
            if ! echo "$GENERIC_APPS" | grep -q "^$label:"; then
                GENERIC_APPS="${GENERIC_APPS}${GENERIC_APPS:+ }$label:$found_path"
                log_success "Generic app ($label) discovered at: $found_path"
            fi
        fi
    done
}

# =============================================================================

# SYMLINK MANAGEMENT
# =============================================================================

update_symlink() {
    local link_path="$1"
    local target_path="$2"
    [[ -L "$link_path" ]] && rm "$link_path"
    [[ -f "$link_path" ]] && rm "$link_path"
    ln -s "$target_path" "$link_path"
    log "Updated symlink: $link_path -> $target_path"
}

# =============================================================================
# VULNERABILITY SCANNING
# =============================================================================

scan_vulnerabilities() {
    local sbom_file="$1"
    local component_name="$2"
    local sbom_dir=$(dirname "$sbom_file")
    local output_file="$sbom_dir/vuln-scan-$component_name-$TIMESTAMP.txt"

    log "Scanning $component_name for vulnerabilities..."
    if grype sbom:"$sbom_file" > "$output_file" 2>&1; then
        local vuln_count
        vuln_count=$(grep "^ " "$output_file" | wc -l | xargs)
        if [[ "$vuln_count" -gt 0 ]]; then
            log_warning "Found $vuln_count potential vulnerabilities in $component_name"
        else
            log_success "No vulnerabilities found in $component_name"
        fi
    else
        log_error "Vulnerability scan failed for $component_name"
        return 1
    fi
    echo "$output_file"
}

# =============================================================================
# COMBINED SBOM AND REPORT GENERATION
# =============================================================================

generate_combined_sbom() {
    log_section "Generating Combined SBOM"
    local combined_file="$SBOM_DIR/sbom-combined-$TIMESTAMP.json"
    
    # Find all SBOM files generated today
    local sbom_files=($(find "$SBOM_DIR" -name "sbom-*-${TIMESTAMP}.json" -type f))
    
    if [[ ${#sbom_files[@]} -eq 0 ]]; then
        log_error "No SBOM files found to combine"
        return 1
    fi
    
    log "Combining ${#sbom_files[@]} SBOM files using jq..."
    
    # Use jq to merge the components arrays from all SBOM files into the first one
    if jq -s '.[0] * {components: [ .[].components // [] | .[] ]}' "${sbom_files[@]}" > "$combined_file"; then
        log_success "Combined SBOM generated: $combined_file"
    else
        log_error "Failed to generate combined SBOM"
        return 1
    fi
    echo "$combined_file"
}

generate_report() {
    log_section "Generating Vulnerability & Dependency Report"
    local report_file="$SBOM_DIR/REPORT-$TIMESTAMP.md"
    local deps_json_file="$SBOM_DIR/deps-outdated-$TIMESTAMP.json"

    echo "$DEPS_DATA_JSON" > "$deps_json_file"

    cat > "$report_file" << EOF
# SBOMaker SBOM Vulnerability & Dependency Report

**Generated:** $(date)
**Project:** SBOMaker
**Schema Version:** CycloneDX $CYCLONEDX_SCHEMA_VERSION

---

## 📊 Summary

### Vulnerabilities
| Component | SBOM File | Vulnerabilities |
|-----------|-----------|-----------------|
| Web UI    | webui/sbom-webui-$TIMESTAMP.json | $(get_vuln_count "webui") |
| Mobile    | mobile/sbom-mobile-$TIMESTAMP.json | $(get_vuln_count "mobile") |
| Docker    | docker/sbom-docker-$TIMESTAMP.json | $(get_vuln_count "docker") |

### Dependency Updates
| Category | Count |
|----------|-------|
| 🔴 Critical (MAJOR) | $CRITICAL_DEPS |
| 🟡 Recommended (MINOR/PATCH) | $RECOMMENDED_DEPS |
| 🔵 Available (All) | $AVAILABLE_DEPS |

---

## 📦 Dependency Updates

${DEPS_DATA_MD:-"No outdated dependencies found."}

---

## 🛡️ Vulnerabilities

### Generic Applications
$(for app in $GENERIC_APPS; do
    local label="${app%%:*}"
    echo "#### $label"
    echo "\`\`\`"
    cat "$SBOM_DIR/generic/$label/vuln-scan-generic-$label-$TIMESTAMP.txt" 2>/dev/null || echo "No scan results available"
    echo "\`\`\`"
    echo ""
done)

### Web UI
\`\`\`
$(cat "$SBOM_DIR/webui/vuln-scan-webui-$TIMESTAMP.txt" 2>/dev/null || echo "No scan results available")
\`\`\`

### Mobile App
\`\`\`
$(cat "$SBOM_DIR/mobile/vuln-scan-mobile-$TIMESTAMP.txt" 2>/dev/null || echo "No scan results available")
\`\`\`

### Docker Image
\`\`\`
$(cat "$SBOM_DIR/docker/vuln-scan-docker-$TIMESTAMP.txt" 2>/dev/null || echo "No scan results available")
\`\`\`

---

## 💡 Recommendations

1. **Address Critical Vulnerabilities**: Review all HIGH and CRITICAL vulnerabilities immediately.
2. **Update Major Dependencies**: Address MAJOR dependency updates to benefit from new features and stability.
3. **Keep Packages Current**: Regularly run this script to stay on top of security patches.

---

*Report generated by SBOMaker.sh*
EOF

    log_success "Report generated: $report_file"
    log_success "Dependency JSON: $deps_json_file"
}

get_vuln_count() {
    local component="$1"
    local vuln_file="$SBOM_DIR/$component/vuln-scan-$component-$TIMESTAMP.txt"
    if [[ -f "$vuln_file" ]]; then
        grep -E "^(CVE|GHSA)-" "$vuln_file" 2>/dev/null | wc -l | xargs
    else
        echo "N/A"
    fi
}

# =============================================================================
# CLEANUP FUNCTIONS
# =============================================================================

cleanup_old_sboms() {
    log_section "Cleanup"
    log "Keeping only the most recent version of each SBOM and scan..."
    
    # 1. Subdirectories in sbom/
    find "$SBOM_DIR" -mindepth 1 -type d | while read -r dir; do
        # Keep newest sbom-*.json
        local files=($(ls -t "$dir"/sbom-*.json 2>/dev/null))
        if [[ ${#files[@]} -gt 1 ]]; then
            for ((i=1; i<${#files[@]}; i++)); do
                rm -f "${files[$i]}"
            done
        fi
        
        # Keep newest vuln-scan-*.txt
        local scans=($(ls -t "$dir"/vuln-scan-*.txt 2>/dev/null))
        if [[ ${#scans[@]} -gt 1 ]]; then
            for ((i=1; i<${#scans[@]}; i++)); do
                rm -f "${scans[$i]}"
            done
        fi
    done
    
    # 2. Root sbom/
    local reports=($(ls -t "$SBOM_DIR"/REPORT-*.md 2>/dev/null))
    if [[ ${#reports[@]} -gt 1 ]]; then
        for ((i=1; i<${#reports[@]}; i++)); do
            rm -f "${reports[$i]}"
        done
    fi
    
    local deps=($(ls -t "$SBOM_DIR"/deps-outdated-*.json 2>/dev/null))
    if [[ ${#deps[@]} -gt 1 ]]; then
        for ((i=1; i<${#deps[@]}; i++)); do
            rm -f "${deps[$i]}"
        done
    fi
    
    log_success "Cleanup complete"
}

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

parse_arguments() {
    if [[ $# -eq 0 ]]; then
        GENERATE_ALL=true
        return
    fi
    GENERATE_WEBUI=false
    GENERATE_MOBILE=false
    GENERATE_DOCKER=false
    GENERATE_ALL=false

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --webui) GENERATE_WEBUI=true; shift ;;
            --mobile) GENERATE_MOBILE=true; shift ;;
            --docker) GENERATE_DOCKER=true; shift ;;
            --all) GENERATE_ALL=true; shift ;;
            --no-fail) NO_FAIL=true; shift ;;
            --docker-image) DOCKER_IMAGE="$2"; shift 2 ;;
            --help|-h) show_help; exit 0 ;;
            *) log_error "Unknown option: $1"; show_help; exit 1 ;;
        esac
    done
}

show_help() {
    cat << EOF
SBOM Update Script for SBOMaker
Usage: $0 [OPTIONS]

Options:
    --webui         Generate SBOM for Web UI only
    --mobile        Generate SBOM for Mobile App only
    --docker        Generate SBOM for Docker Image only
    --all           Generate SBOM for all components (default)
    --no-fail        Do not exit with error if vulnerabilities/deps found
    --docker-image  Specify Docker image name (must be specified via --docker-image)
    --help, -h      Show this help message

Examples:
    $0                  # Generate all SBOMs
    $0 --webui          # Generate only Web UI SBOM
    $0 --mobile --docker # Generate Mobile and Docker SBOMs
    $0 --docker-image myapp:v1.0 --docker  # Generate Docker SBOM for specific image
    $0 --no-fail        # Run everything but always exit 0
EOF
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    SBOMaker                                  ║"
    echo "║     Software Bill of Materials Generator & Scanner           ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    parse_arguments "$@"
    load_config
    init_log
    discover_components
    log "Starting SBOM generation..."
    check_dependencies
    mkdir -p "$SBOM_DIR"

    local generated_files=()
    local vuln_scan_files=()

    if [[ "$GENERATE_ALL" == true ]] || [[ "$GENERATE_WEBUI" == true ]]; then
        local webui_sbom=$(generate_webui_sbom)
        if [[ -n "$webui_sbom" ]]; then
            generated_files+=("$webui_sbom")
            local vuln_scan=$(scan_vulnerabilities "$webui_sbom" "webui")
            [[ -n "$vuln_scan" ]] && vuln_scan_files+=("$vuln_scan")
        fi
    fi

    if [[ "$GENERATE_ALL" == true ]] || [[ "$GENERATE_MOBILE" == true ]]; then
        local mobile_sbom=$(generate_mobile_sbom)
        if [[ -n "$mobile_sbom" ]]; then
            generated_files+=("$mobile_sbom")
            local vuln_scan=$(scan_vulnerabilities "$mobile_sbom" "mobile")
            [[ -n "$vuln_scan" ]] && vuln_scan_files+=("$vuln_scan")
        fi
    fi

    if [[ "$GENERATE_ALL" == true ]] || [[ "$GENERATE_DOCKER" == true ]]; then
        local docker_sbom=$(generate_docker_sbom)
        if [[ -n "$docker_sbom" ]]; then
            generated_files+=("$docker_sbom")
            local vuln_scan=$(scan_vulnerabilities "$docker_sbom" "docker")
            [[ -n "$vuln_scan" ]] && vuln_scan_files+=("$vuln_scan")
        fi
    fi

    if [[ -n "$GENERIC_APPS" ]]; then
        for app in $GENERIC_APPS; do
            local label="${app%%:*}"
            local path="${app#*:}"
            local generic_sbom=$(generate_generic_sbom "$label" "$path")
            if [[ -n "$generic_sbom" ]]; then
                generated_files+=("$generic_sbom")
                local vuln_scan=$(scan_vulnerabilities "$generic_sbom" "generic-$label")
                [[ -n "$vuln_scan" ]] && vuln_scan_files+=("$vuln_scan")
            fi
        done
    fi

    if [[ ${#generated_files[@]} -gt 1 ]]; then
        generate_combined_sbom
    fi

    if [[ ${#vuln_scan_files[@]} -gt 0 ]]; then
        generate_report
    fi

    cleanup_old_sboms

    log_section "Summary"
    log_success "SBOM generation complete!"
    log "Generated ${#generated_files[@]} SBOM file(s)"

    # EXIT LOGIC
    if [[ "$NO_FAIL" == "false" ]]; then
        if [[ $CRITICAL_DEPS -gt 0 ]]; then
            log_error "CRITICAL: $CRITICAL_DEPS major dependency updates required!"
            exit 1
        fi
    fi

    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  SBOM generation completed successfully!${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "Generated files:"
    for file in "${generated_files[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo "View the report:"
    echo "  cat $SBOM_DIR/REPORT-$TIMESTAMP.md"
    echo ""
}

main "$@"
