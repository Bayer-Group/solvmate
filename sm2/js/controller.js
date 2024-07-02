const update_view = function (res) {
    if (res === undefined) return;
    console.log("Request complete! response:" + res);

    var all_cells = "";
    res.forEach(row => {
        all_cells += `
      <div>
        <div class="mol">${row['solvent SMILES']}</div>
        <div>${row['log S']}</div>
      </div>
      `;
    });

    document.getElementById("main-content").innerHTML += all_cells;
    draw_molecules();
    document.getElementById("status-message").innerText = "";
};
const update_plot_view = function (res) {
    if (res === undefined) return;
    console.log("Request complete! response:" + res);
    document.getElementById("main-content").innerHTML += res["svg"];
    draw_molecules();
    document.getElementById("status-message").innerText = "";
};
var API_SMILES = "";
const update_model = function () {
    document.getElementById("status-message").innerText = "calculating ...";

    fetch("/plot-rank-by-solubility", {
        method: "POST",
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ "solute SMILES": API_SMILES, "solvents": get_selected_solvents(), })
    }).then(res => res.json()).then(
        res => update_plot_view(res)
    );
};