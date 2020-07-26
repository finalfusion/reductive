{ pkgs ? import (import ./nix/sources.nix).nixpkgs {}

# Build finalfusion-utils with OpenBLAS tests.
, withOpenblas ? false
}:

let
  sources = import ./nix/sources.nix;
  crateTools = pkgs.callPackage "${sources.crate2nix}/tools.nix" {};
  cargoNix = pkgs.callPackage (crateTools.generatedCargoNix {
    name = "finalfusion";
    src = pkgs.nix-gitignore.gitignoreSource [ ".git/" "nix/" "*.nix" "result*" ] ./.;
  }) {
    inherit buildRustCrate;

    rootFeatures = with pkgs; [ "default" ]
      ++ lib.optional withOpenblas "openblas-test";
  };
  crateOverrides = with pkgs; defaultCrateOverrides // {
    openblas-src = attr: {
      nativeBuildInputs = [ perl ];
    };

    reductive = attr: {
      buildInputs = lib.optionals withOpenblas [ gfortran.cc.lib openblas ]
        ++ lib.optionals stdenv.isDarwin [ darwin.Security ];
    };
  };
  buildRustCrate = pkgs.buildRustCrate.override {
    defaultCrateOverrides = crateOverrides;
  };
in cargoNix.rootCrate.build.override { runTests = true; }
