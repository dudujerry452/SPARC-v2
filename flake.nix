{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      libPath = "${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:/usr/lib/wsl/lib";
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        name = "sparc-env";
        buildInputs = with pkgs; [
          python3
          python3Packages.opencv4
          python3Packages.scikit-image
          stdenv.cc.cc.lib
          zlib
          git
        ];

        shellHook = ''
          export LD_LIBRARY_PATH="${libPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

          # Recreate venv if it doesn't exist or lacks system-site-packages
          if [ ! -d .venv ] || ! grep -q "include-system-site-packages = true" .venv/pyvenv.cfg 2>/dev/null; then
            echo "Creating Python venv with system-site-packages..."
            rm -rf .venv
            ${pkgs.python3}/bin/python -m venv .venv --system-site-packages
          fi
          source .venv/bin/activate

          check() {
            if ! python -c "import torch" 2>/dev/null; then
              echo "Installing PyTorch (cu118)..."
              pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
            fi

            if ! python -c "import scipy, cv2, tifffile" 2>/dev/null; then
              echo "Installing Python dependencies..."
              pip install numpy scipy pillow tifffile
            fi
          }

          echo "SPARC environment ready."
          echo "Python: $(python --version)"
          echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
        '';
      };
    };
}
