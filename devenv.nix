{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: let
  # UV works best with hardlinks to its cache, so we try to find a cache directory on the same filesystem as the virtual environment.
  findUvCacheDir = ''
    MOUNT_POINT=$(${pkgs.coreutils}/bin/df -P "${config.env.DEVENV_STATE}/venv" | tail -1 | ${pkgs.gawk}/bin/awk '{print $6}')
    # set the cache directory for uv to be on the same filesystem as the current working directory
    # This makes sure that cache files are hardlinked to the virtual environment and not copied

    # Try a shared cache directory first
    if [ -d "$MOUNT_POINT/.cache" ] && [ -w "$MOUNT_POINT/.cache" ]; then
      UV_CACHE_DIR="$MOUNT_POINT/.cache/uv"
    # Fallback to user-specific cache directory
    elif [ -d "$MOUNT_POINT/$USER/.cache" ] && [ -w "$MOUNT_POINT/$USER/.cache" ]; then
      UV_CACHE_DIR="$MOUNT_POINT/$USER/.cache/uv"
    fi
    export UV_CACHE_DIR
  '';
  # Selects the CUDA toolkit version 12.3 from the available CUDA packages in Nixpkgs.
  # The `cudaPackages` attribute provides access to the full suite of CUDA tools and libraries for this version.
  # To use a different CUDA version, replace `cudaPackages_12_3` with another version (e.g., `cudaPackages_11_8`).
  # Individual components (such as `cuda`, `cudatoolkit`, `cudnn`, etc.) can be accessed as attributes of `cudaPackages`
  # and added to the `packages` list in your devenv shell as needed.
  # Example: `cudaPackages.cudatoolkit`, `cudaPackages.cudnn`, etc.
  cudaPackages = pkgs.cudaPackages_12_4;
  # cudaPackages = pkgs.cudaPackages_10_0
  # cudaPackages = pkgs.cudaPackages_10_1
  # cudaPackages = pkgs.cudaPackages_10_2
  # cudaPackages = pkgs.cudaPackages_10
  # cudaPackages = pkgs.cudaPackages_11_0
  # cudaPackages = pkgs.cudaPackages_11_1
  # cudaPackages = pkgs.cudaPackages_11_2
  # cudaPackages = pkgs.cudaPackages_11_3
  # cudaPackages = pkgs.cudaPackages_11_4
  # cudaPackages = pkgs.cudaPackages_11_5
  # cudaPackages = pkgs.cudaPackages_11_6
  # cudaPackages = pkgs.cudaPackages_11
in {
  imports = [
    inputs.devenv-templates.devenvModules.default
  ];

  name = "minimal-uv";
  # https://devenv.sh/basics/
  env = {
    GREET = "devenv";
    UV_PROJECT = "${config.devenv.root}";
    # Environment variables for building packages like Pillow
    CPPFLAGS = "-I${pkgs.zlib.dev}/include -I${pkgs.libjpeg.dev}/include -I${pkgs.libpng.dev}/include -I${pkgs.libtiff.dev}/include -I${pkgs.libwebp}/include -I${pkgs.freetype.dev}/include -I${pkgs.lcms2.dev}/include -I${pkgs.openjpeg}/include";
    LDFLAGS = "-L${pkgs.zlib}/lib -L${pkgs.libjpeg}/lib -L${pkgs.libpng}/lib -L${pkgs.libtiff}/lib -L${pkgs.libwebp}/lib -L${pkgs.freetype}/lib -L${pkgs.lcms2}/lib -L${pkgs.openjpeg}/lib";
    PKG_CONFIG_PATH = "${pkgs.zlib.dev}/lib/pkgconfig:${pkgs.libjpeg.dev}/lib/pkgconfig:${pkgs.libpng.dev}/lib/pkgconfig:${pkgs.libtiff.dev}/lib/pkgconfig:${pkgs.libwebp}/lib/pkgconfig:${pkgs.freetype.dev}/lib/pkgconfig:${pkgs.lcms2.dev}/lib/pkgconfig:${pkgs.openjpeg}/lib/pkgconfig";
  };

  generateEnvrcLocal = true;

  # Uncomment below to show a menu upon entering the shell.
  # menu = {
  #   enable = true;
  #   showInstalledPackages = true;
  #   showPreCommitHooks = true;
  # };

  # https://devenv.sh/packages/
  packages = with pkgs; [
    # cudaPackages.cuda_nvcc
    alejandra
    git
    # System dependencies for building Python packages like Pillow
    zlib.dev
    libjpeg.dev
    libpng.dev
    libtiff.dev
    libwebp
    freetype.dev
    lcms2.dev
    openjpeg
    pkg-config
  ];

  # https://devenv.sh/languages/
  languages.python = {
    enable = true;
    version = "3.11";
    venv.enable = true;
    uv = {
      enable = true;
      sync.enable = true;
    };
    manylinux.enable = false;
    libraries = with pkgs; [
      zlib
      libjpeg
      libpng
      libtiff
      libwebp
      freetype
      lcms2
      openjpeg
    ];
  };

  # https://devenv.sh/processes/
  # processes.cargo-watch.exec = "cargo-watch";

  # https://devenv.sh/services/
  # services.postgres.enable = true;

  # https://devenv.sh/scripts/
  scripts = {
    reinstall-venv.exec = "rm -rf $DEVENV_ROOT/.venv $DEVENV_ROOT/.direnv $DEVENV_ROOT/.devenv && direnv reload";
  };

  enterShell = ''
    # Just remove the line below if you don't want to see the cow
    ${pkgs.cowsay}/bin/cowsay "If you have any questions about installing packages, please check the _README.md file in this directory! (Also if you want to get rid of the cow ;P)" | ${pkgs.lolcat}/bin/lolcat 2> /dev/null
  '';

  containers.shell.defaultCopyArgs = [
    "--dest-daemon-host=unix:///run/user/1000/podman/podman.sock"
  ];

  # https://devenv.sh/tasks/
  tasks = {
    "devenv:python:set-uv-cache" = {
      description = "Find UV cache on same filesystem.";
      exec = findUvCacheDir;
      exports = ["UV_CACHE_DIR"];
      before = ["devenv:python:uv"];
    };
  };

  # https://devenv.sh/tests/
  # enterTest = ''
  #   echo "Running tests"
  #   git --version | grep --color=auto "${pkgs.git.version}"
  # '';

  # https://devenv.sh/git-hooks/
  git-hooks.hooks = {
    shellcheck.enable = true;
    alejandra.enable = true;
    ruff.enable = true;
    ruff-format.enable = true;
  };
  # See full reference at https://devenv.sh/reference/options/
}
