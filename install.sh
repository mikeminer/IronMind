#!/usr/bin/env sh
set -eu

repo="mikeminer/IronMind"
branch="main"
install_dir="${IRONMIND_INSTALL_DIR:-$HOME/.local/share/ironmind}"
bin_dir="${HOME}/.local/bin"
tmp_dir="$(mktemp -d)"
zip_path="${tmp_dir}/ironmind.zip"

cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

if ! command -v node >/dev/null 2>&1; then
  echo "Node.js 18+ is required. Install Node.js, then run this installer again." >&2
  exit 1
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "warning: Ollama was not found. Install it from https://ollama.com/download, then run: ollama pull qwen3-coder:30b" >&2
fi

mkdir -p "$install_dir" "$bin_dir"

if command -v curl >/dev/null 2>&1; then
  curl -fsSL "https://github.com/${repo}/archive/refs/heads/${branch}.zip" -o "$zip_path"
elif command -v wget >/dev/null 2>&1; then
  wget -q "https://github.com/${repo}/archive/refs/heads/${branch}.zip" -O "$zip_path"
else
  echo "curl or wget is required." >&2
  exit 1
fi

if command -v unzip >/dev/null 2>&1; then
  unzip -q "$zip_path" -d "$tmp_dir"
else
  echo "unzip is required." >&2
  exit 1
fi

src_dir="$(find "$tmp_dir" -maxdepth 1 -type d -name 'IronMind-*' | head -n 1)"
if [ -z "$src_dir" ]; then
  echo "Downloaded archive did not contain the IronMind source directory." >&2
  exit 1
fi

rm -rf "$install_dir"
mkdir -p "$install_dir"
cp -R "$src_dir"/. "$install_dir"/

cat > "$bin_dir/ironmind" <<EOF
#!/usr/bin/env sh
exec node "$install_dir/bin/ironmind.mjs" "\$@"
EOF
chmod +x "$bin_dir/ironmind"

mkdir -p "$HOME/.ironmind"
if [ ! -f "$HOME/.ironmind/ironmind.json" ]; then
  cat > "$HOME/.ironmind/ironmind.json" <<EOF
{
  "model": "qwen3-coder:30b",
  "context": 131072,
  "kvDiskDir": "$HOME/.ironmind/kvcache",
  "kvDiskSpaceMb": 16384,
  "ollamaUrl": "http://127.0.0.1:11434"
}
EOF
fi

echo "IronMind installed."
echo "Run: ironmind"
echo "If needed: ollama pull qwen3-coder:30b"
echo "Chatbot: http://127.0.0.1:4141"
