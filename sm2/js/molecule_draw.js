const draw_molecules = function () {
    Array.from(document.getElementsByClassName("mol")).forEach((element) => {
        var smiles = element.innerHTML;
        var mol = window.RDKit.get_mol(smiles);
        var dest = element;
        var svg = mol.get_svg();
        dest.innerHTML = "<div class='drawing'>" + svg + "</div>";
    });
};
window
    .initRDKitModule()
    .then(function (RDKit) {
        console.log("RDKit version: " + RDKit.version());
        window.RDKit = RDKit;
        draw_molecules();
        /**
         * The RDKit module is now loaded.
         * You can use it anywhere.
         */
    })
    .catch(() => {
        // handle loading errors here...
    });