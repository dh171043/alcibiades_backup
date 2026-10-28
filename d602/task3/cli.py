from datetime import datetime

import click


@click.command()
@click.option(
    "--airport", required=True, type=str, help="Arrival airport code (EX: IND)."
)
@click.option(
    "--dep-time", required=True, type=str, help="Local departure time (EX: HH:MM:SS)."
)
@click.option(
    "--arr-time", required=True, type=str, help="Local arrival time (EX: HH:MM:SS)."
)
def main(airport: str, dep_time: str, arr_time: str):
    click.echo("--- Running Airline Delay Prediction CLI  ---")

    from predictor import load_model_and_encodings, make_prediction

    try:
        departure_time_obj = datetime.strptime(dep_time, "%H:%M:%S").time()
        arrival_time_obj = datetime.strptime(arr_time, "%H:%M:%S").time()
    except ValueError:
        click.secho("Error: Invalid time format, use HH:MM:SS", fg="red")
    except Exception as e:
        click.secho(
            f"DEBUG: An unexpected error occured: {type(e).__name__}", fg="yellow"
        )
        click.secho(f"Error details: {e}", fg="red")
        exit(1)

    click.echo(">> Loading model and encodings")
    model, airport_encodings = load_model_and_encodings()
    click.echo(">> Assets Loaded Successfully")

    try:
        click.echo(">> Making prediction...")
        prediction_minutes = make_prediction(
            model=model,
            airport_encodings=airport_encodings,
            arrival_airport=airport,
            departure_time=departure_time_obj,
            arrival_time=arrival_time_obj,
        )
    except KeyError as e:
        click.secho(f"Error: {e}", fg="red")
        exit(1)
    click.echo(">> Prediction successful!")
    click.secho(
        f"Predicted departure delay for a flight arriving at {airport.upper()} is: {prediction_minutes:.2f} minutes.",
        fg="green",
        bold=True,
    )


if __name__ == "__main__":
    main()
