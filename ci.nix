{ pkgs ? import (import ./nix/sources.nix).nixpkgs {} }:

let
  finalfusion = pkgs.callPackage ./default.nix {};
in [
  finalfusion
  (finalfusion.override { withOpenblas = true; })
]
