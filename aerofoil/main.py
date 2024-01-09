import typer

from aerofoil.commands.clean_image import cleanimg
from aerofoil.commands.train_hybrid import trainhybrid
from aerofoil.commands.predict_image import predictimg
from aerofoil.commands.train_image import trainimg

app = typer.Typer(help="Aplikasi CLI untuk keperluan pelatihan model CNN.")

app.command(
    help="Fungsi untuk membersihkan images yang tidak ada dalam out.csv. Hasil disimpan dalam folder out (default)."
)(cleanimg)

app.command(help="Fungsi untuk melatih model AI dengan CNN.")(trainimg)
app.command(help="Fungsi untuk prediksi airfoil.")(predictimg)
app.command(help="Fungsi untuk melatih model AI dengan CNN hybrid.")(trainhybrid)
