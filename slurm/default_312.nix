# default_312.nix — Python 3.12.11 dev shell (manual venv management)

with import (builtins.fetchGit {
  url = "https://github.com/NixOS/nixpkgs";
  rev = "8cbadfa068534bdd8238eea362d2bf0b1d46b7e8"; # commit with Python 3.12.11
}) { config.allowUnfree = true; };

let
  py      = pkgs.python312; # pinned interpreter
  libPath = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc pkgs.zlib ];
in
pkgs.mkShell {
  buildInputs = [
    py
    pkgs.stdenv.cc.cc
    pkgs.git
  ] ++ (import ./system-dependencies.nix { inherit pkgs; });

  shellHook = ''
    set -e

    # use local tmpdir to avoid /tmp/env-vars permission errors
    export TMPDIR="$PWD/.nix-tmp"
    mkdir -p "$TMPDIR"

    export LD_LIBRARY_PATH="${libPath}:$LD_LIBRARY_PATH"

    VENV="$PWD/_venv"

    # create venv with the pinned Python if missing or wrong version
    if [ ! -x "$VENV/bin/python" ] || ! "$VENV/bin/python" -c 'import sys; exit(0 if sys.version.startswith("3.12.") else 1)'; then
      echo "[venv] creating new venv with ${py}/bin/python ..."
      rm -rf "$VENV"
      ${py}/bin/python -m venv "$VENV"
    fi

    # activate it
    . "$VENV/bin/activate"

    echo "[venv] using $(python -V)"

    # install requirements if needed
    if [ -f requirements.txt ]; then
      HASH_FILE="$VENV/.req-hash"
      NEW_HASH=$(sha256sum requirements.txt | awk '{print $1}')
      OLD_HASH=$(cat "$HASH_FILE" 2>/dev/null || true)

      if [ "$NEW_HASH" != "$OLD_HASH" ]; then
        echo "[venv] installing requirements.txt ..."
        python -m pip install --upgrade pip wheel
        python -m pip install -r requirements.txt
        echo "$NEW_HASH" > "$HASH_FILE"
      fi
    fi
  '';
}
