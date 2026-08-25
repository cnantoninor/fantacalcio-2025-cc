"""CLI di FANTABUSTE. Fase 0: solo scaffolding — i comandi reali (prepara,
tornata, chiudi-buste) arrivano con il Modulo E in Fase 2, vedi docs/DESIGN.md.
"""

from __future__ import annotations

import argparse

from fantabuste import fixtures


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="fantabuste",
        description="Supporto decisionale per l'asta a buste chiuse del Fantacalcio 2026/27.",
    )
    sub = parser.add_subparsers(dest="comando")

    sub.add_parser(
        "genera-fixture",
        help="Rigenera le fixture sintetiche in data/fixtures/ (riproducibile da seed).",
    )

    args = parser.parse_args()

    if args.comando == "genera-fixture":
        fixtures.main()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
