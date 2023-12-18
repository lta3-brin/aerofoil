import typer

from aerofoil.commands.clean_image import cleanimg

app = typer.Typer(help="Aplikasi CLI untuk keperluan pelatihan model CNN.")

app.command(
    help="Fungsi untuk membersihkan images yang tidak ada dalam out.csv. Hasil disimpan dalam folder out (default)."
)(cleanimg)
