"""C128 OPA-PFW entry using the common OPA Click options and configuration."""
import click
from train_overlap import main as overlap


@click.command()
@click.option("--foreground-cache", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--w-max", type=click.FloatRange(min=1), default=2.0)
def main(foreground_cache, w_max, **kwargs):
    required = dict(precond="pedm", backbone="songunet", overlap_mode="independent",
                    overlap_alignment="symmetric", overlap_geometry="identical",
                    overlap_spatial_weight="uniform", augment=0, cbase=128, lambda_overlap=1.25)
    for key, value in required.items():
        if kwargs[key] != value:
            raise click.UsageError(f"C128 OPA1p25 requires {key}={value}")
    overlap.callback(pfw_cache=foreground_cache, pfw_w_max=w_max, **kwargs)


main.params.extend(overlap.params)

if __name__ == "__main__":
    main()
