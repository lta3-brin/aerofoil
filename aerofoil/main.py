import typer

app = typer.Typer(help="Aplikasi CLI untuk keperluan pelatihan model CNN.")


@app.command()
def shoot():
    """
    Shoot the portal gun
    """
    typer.echo("Shooting portal gun")


@app.command()
def load():
    """
    Load the portal gun
    """
    typer.echo("Loading portal gun")
