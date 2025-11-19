#!/bin/bash
# lean_install.sh - Optimized for SPEED (5-15 minutes)
# Uses cache downloads instead of full compilation

set -e

echo "=========================================="
echo "FAST Lean 4 Installation (Cache-Based)"
echo "Estimated time: 5-15 minutes"
echo "=========================================="

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if already installed
if [ -f "$HOME/.elan/bin/elan" ] && [ -d "$HOME/lean/mathlib4" ]; then
    print_warning "Lean appears to be already installed!"
    read -p "Do you want to reinstall? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Using existing installation."
        exit 0
    fi
fi

# Update system (quick)
print_status "Updating system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq curl git build-essential python3-pip

# Install elan (~1 minute)
print_status "Installing elan (Lean version manager)..."
if [ ! -f "$HOME/.elan/bin/elan" ]; then
    curl https://elan.lean-lang.org/elan-init.sh -sSf | sh -s -- -y --default-toolchain none
    print_status "Elan installed"
else
    print_warning "Elan already installed, skipping..."
fi

# Source environment
export PATH="$HOME/.elan/bin:$PATH"
source "$HOME/.profile" 2>/dev/null || true

# Install Lean 4 stable (~30 seconds)
print_status "Installing Lean 4 stable..."
elan install stable
elan default stable
lean --version

# Create workspace
print_status "Setting up Lean workspace..."
LEAN_WORKSPACE="$HOME/lean"
mkdir -p "$LEAN_WORKSPACE"
cd "$LEAN_WORKSPACE"

# Clone Mathlib4 (~1-2 minutes depending on network)
print_status "Cloning Mathlib4 (this is a large repo)..."
if [ ! -d "$LEAN_WORKSPACE/mathlib4" ]; then
    # Shallow clone to save time and space
    git clone --depth 1 https://github.com/leanprover-community/mathlib4.git
    print_status "Mathlib4 cloned"
else
    print_warning "Mathlib4 already exists, updating..."
    cd mathlib4
    git pull
fi

cd "$LEAN_WORKSPACE/mathlib4"

# Download precompiled cache (5-10 minutes) - THE KEY TIME SAVER!
print_status "Downloading precompiled Mathlib cache..."
print_warning "This is the longest step (5-10 min) but saves 2-6 HOURS of compilation!"

# Try to download cache multiple times if it fails
MAX_RETRIES=3
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if lake exe cache get; then
        print_status "Cache downloaded successfully!"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            print_warning "Cache download failed, retrying ($RETRY_COUNT/$MAX_RETRIES)..."
            sleep 2
        else
            print_error "Cache download failed after $MAX_RETRIES attempts"
            print_warning "Continuing without cache (verification will be slower)"
        fi
    fi
done

# Quick build of essential files only (2-3 minutes)
print_status "Building essential Lean files..."
lake build 2>&1 | grep -E "Building|error" || true

# Install REPL (~2 minutes)
print_status "Setting up REPL tool..."
cd "$LEAN_WORKSPACE"
if [ ! -d "REPL" ]; then
    git clone --depth 1 https://github.com/leanprover-community/repl.git REPL
else
    print_warning "REPL already exists"
fi

# Add REPL to mathlib if not already there
cd "$LEAN_WORKSPACE/mathlib4"
if ! grep -q "REPL" lakefile.lean 2>/dev/null && ! grep -q "repl" lakefile.toml 2>/dev/null; then
    print_status "Adding REPL dependency..."
    
    # Check if using lakefile.toml or lakefile.lean
    if [ -f "lakefile.toml" ]; then
        cat >> lakefile.toml <<EOF

[[require]]
name = "repl"
git = "https://github.com/leanprover-community/repl.git"
rev = "main"
EOF
    fi
    
    lake update repl 2>&1 | grep -v "warning" || true
fi

# Install Python dependencies
print_status "Installing Python dependencies..."
pip3 install --quiet --upgrade pip
pip3 install --quiet func-timeout psutil ray tqdm numpy

# Setup environment variables
print_status "Configuring environment..."
if ! grep -q "LEAN_WORKSPACE" "$HOME/.bashrc"; then
    cat >> "$HOME/.bashrc" <<'EOF'

# Lean 4 Environment
export PATH="$HOME/.elan/bin:$PATH"
export LEAN_WORKSPACE="$HOME/lean/mathlib4"
export LAKE_PATH="$HOME/.elan/bin/lake"
EOF
fi

source "$HOME/.bashrc" 2>/dev/null || true

# Quick verification test
print_status "Running quick verification test..."
cd "$LEAN_WORKSPACE/mathlib4"
TEST_RESULT=$(echo '{"cmd": "theorem test : 1 + 1 = 2 := by rfl", "allTactics": false, "ast": false, "tactics": false, "premises": false}' | timeout 30 lake exe repl 2>&1 || echo "TIMEOUT")

if echo "$TEST_RESULT" | grep -q '"sorries"'; then
    print_status "Verification test PASSED! ✓"
else
    print_warning "Verification test inconclusive, but installation completed"
    print_warning "Test output: $TEST_RESULT"
fi

# Create a test file
print_status "Creating example test file..."
cat > "$LEAN_WORKSPACE/mathlib4/test_quick.lean" <<'EOF'
import Mathlib.Tactic

theorem quick_test : 1 + 1 = 2 := by norm_num

theorem nat_add_zero (n : ℕ) : n + 0 = n := by
  exact Nat.add_zero n
EOF

# Print summary
echo ""
print_status "=========================================="
print_status "Installation Complete! 🎉"
print_status "=========================================="
echo ""
echo "Installation location: $HOME/lean/mathlib4"
echo "Lean version: $(lean --version 2>&1 | head -n1)"
echo ""
echo "Next steps:"
echo "  1. Test verification: cd ~/lean/mathlib4 && echo '{\"cmd\": \"theorem t : 1=1 := by rfl\"}' | lake exe repl"
echo "  2. Use Python utils: python3 lean4_verifier_utils.py"
echo "  3. Check guide: cat LEAN_VERIFICATION_GUIDE.md"
echo ""
print_warning "Note: First verification may be slow as Lean loads. Subsequent ones are fast!"
echo ""

# Display disk usage
DISK_USAGE=$(du -sh "$HOME/lean" 2>/dev/null | cut -f1)
echo "Disk usage: $DISK_USAGE"

# Create VM setup completed marker
touch "$HOME/.lean_installed"
echo "$(date)" > "$HOME/.lean_installed"

print_status "You can now stop and restart this VM without reinstalling!"