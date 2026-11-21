import os
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt


def format_date(date_str):
    """'20260701' -> '2026. 07. 01.'"""
    if len(date_str) == 8:
        return f"{date_str[:4]}. {date_str[4:6]}. {date_str[6:8]}."
    return date_str


def compute_shared_extent(npz_files, buffer_deg=1.5):
    """Kiszámítja az összes megadott .npz fájl közös befoglaló koordinátáit."""
    global_min_lon, global_max_lon = np.inf, -np.inf
    global_min_lat, global_max_lat = np.inf, -np.inf

    for f in npz_files:
        data = np.load(f, allow_pickle=True)
        all_lons = np.concatenate(data["lons"])
        all_lats = np.concatenate(data["lats"])

        global_min_lon = min(global_min_lon, all_lons.min())
        global_max_lon = max(global_max_lon, all_lons.max())
        global_min_lat = min(global_min_lat, all_lats.min())
        global_max_lat = max(global_max_lat, all_lats.max())

    return [
        np.floor(global_min_lon - buffer_deg),
        np.ceil(global_max_lon + buffer_deg),
        np.floor(global_min_lat - buffer_deg),
        np.ceil(global_max_lat + buffer_deg),
    ]


def plot_six_panel(
    area,
    dates,
    models=("aifs", "ifs"),
    lead_time=168,
    isobar=500,
    spat_res="0.005",
    manual_extent=None,
    out_dir="uniform_plots",
    fig_width=10.5,
    dpi=300,
):
    os.makedirs(out_dir, exist_ok=True)
    res_raw = spat_res.replace(".", "")

    # 1. Fájlútvonalak összegyűjtése (3 sor = dátumok, 2 oszlop = modellek)
    file_grid = []
    all_files = []
    for date_str in dates:
        row_files = []
        for model in models:
            fpath = f"../traj_sols/{area}/trajectory_solution_{lead_time}h_{isobar}hPa_{date_str}_{model}_{res_raw}.npz"
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Hiányzó fájl: {fpath}")
            row_files.append(fpath)
            all_files.append(fpath)
        file_grid.append(row_files)

    # 2. Közös térképkivágat mind a 6 panelhez
    if manual_extent is not None:
        extent = manual_extent
    else:
        extent = compute_shared_extent(all_files, buffer_deg=1.5)

    min_lon, max_lon, min_lat, max_lat = extent
    print(f"Mind a 6 panelen használt kivágat [min_lon, max_lon, min_lat, max_lat]: {extent}")

    nrows = len(dates)   # 3 sor
    ncols = len(models)  # 2 oszlop

    # 3. Szoros elrendezés és pontos képmagasság számítása (üres fehér sávok ellen)
    left = 0.065
    right = 0.985
    bottom = 0.045
    top = 0.85
    wspace = 0.035  # vízszintes térköz a két oszlop között (kicsi!)
    hspace = 0.14   # függőleges térköz a sorok között (pont elfér a panelcím)

    lon_span = max_lon - min_lon
    lat_span = max_lat - min_lat

    avail_w = fig_width * (right - left)
    ax_w = avail_w / (ncols + (ncols - 1) * wspace)
    ax_h = ax_w * (lat_span / lon_span)
    avail_h = ax_h * (nrows + (nrows - 1) * hspace)
    fig_height = avail_h / (top - bottom)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    fig.subplots_adjust(
        left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace
    )

    panel_letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    legend_handles = None
    legend_labels = None

    for r_idx, date_str in enumerate(dates):
        for c_idx, model in enumerate(models):
            ax = axes[r_idx, c_idx]
            fpath = file_grid[r_idx][c_idx]
            data = np.load(fpath, allow_pickle=True)
            lons_list = data["lons"]
            lats_list = data["lats"]

            ax.set_extent(extent, crs=ccrs.PlateCarree())

            # Földrajzi háttér
            ax.add_feature(cfeature.LAND, facecolor="#f5f5f5", zorder=0)
            ax.add_feature(cfeature.OCEAN, facecolor="#e0f2fe", zorder=0)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="#333333", zorder=1)
            ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.5, zorder=1)

            # Rácsvonalak: szélességi fok csak a bal szélen, hosszúsági csak alul
            gl = ax.gridlines(
                draw_labels=True, linewidth=0.5, color="gray", alpha=0.55, linestyle="--", zorder=2
            )
            gl.top_labels = False
            gl.right_labels = False
            gl.left_labels = (c_idx == 0)
            gl.bottom_labels = (r_idx == nrows - 1)
            gl.x_inline = False
            gl.y_inline = False
            gl.xlabel_style = {"size": 9.5}
            gl.ylabel_style = {"size": 9.5}

            # Kezdeti és végső fonal
            h1, = ax.plot(
                lons_list[0], lats_list[0],
                transform=ccrs.PlateCarree(), color="green", linewidth=2, label="Kezdeti fonal"
            )
            ax.scatter(
                lons_list[0], lats_list[0],
                transform=ccrs.PlateCarree(), color="darkgreen", s=8, zorder=5
            )
            h2, = ax.plot(
                lons_list[-1], lats_list[-1],
                transform=ccrs.PlateCarree(), color="crimson", linewidth=2, label="Végső fonal"
            )
            ax.scatter(
                lons_list[-1], lats_list[-1],
                transform=ccrs.PlateCarree(), color="darkred", s=8, zorder=5
            )

            if legend_handles is None:
                legend_handles = [h1, h2]
                legend_labels = ["Kezdeti fonal", "Végső fonal"]

            # Rövid, jól olvasható panelcím
            p_idx = r_idx * ncols + c_idx
            ax.set_title(
                f"{panel_letters[p_idx]} ECMWF-{model.upper()} – {format_date(date_str)}",
                fontsize=10.5,
                fontweight="bold",
                pad=5,
            )

    # Közös főcím és egyetlen közös jelmagyarázat az ábra tetején
    #fig.suptitle(f"Részecske trajektóriák összehasonlítása ({isobar} hPa, {lead_time} h, {spat_res}°)",fontsize=13,fontweight="bold",y=0.97,)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.942),
        ncol=2,
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        fontsize=10,)
    out_path = os.path.join(
        out_dir, f"panel6_{area}_{isobar}hPa_{lead_time}h_{res_raw}.png"
    )
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    # Add meg a 3 összehasonlítandó dátumot (3 sor x 2 oszlop = 6 panel):
    selected_dates = ["20260601", "20260701", "20261001"]

    saved = plot_six_panel(
        area="hun",
        dates=selected_dates,
        models=("aifs", "ifs"),
        lead_time=168,
        isobar=500,
        spat_res="0.005",
        out_dir="uniform_plots",
        fig_width=10.5,
        dpi=300,
    )
    print(f"Elkészült a 6 paneles ábra: {saved}")
