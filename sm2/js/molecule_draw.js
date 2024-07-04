/* 
    A pure-js implementation of molecule displaying functionality.
    Inline display of molecules relies on rdkit.js SVG images,
    while editing uses the statically bundled Ketcher instance.
*/

const draw_molecules = function () {
    Array.from(document.getElementsByClassName("mol")).forEach((element) => {
        var smiles = element.innerHTML;
        var mol = window.RDKit.get_mol(smiles);
        if (mol) {
            var dest = element;
            var svg = mol.get_svg();
            dest.innerHTML = "<div class='drawing'>" + svg + "</div>";
        }
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

const getKetcher = function (ketcherId) {

    var ketcherFrame = document.getElementById(ketcherId);
    var ketcher = null;

    if ('contentDocument' in ketcherFrame)
        ketcher = ketcherFrame.contentWindow.ketcher;
    else // IE7
        ketcher = document.frames[ketcherId].window.ketcher;

    return ketcher;
}

const currentSmiles = async function () {
    const ketcher = getKetcher('ifKetcher');
    const smi = await ketcher.getSmiles();
    return smi;
};

const runRanking = function () {
    const ketcher = getKetcher('ifKetcher');
    if (ketcher) {
        ketcher.getSmiles().then(
            smiles => {
                API_SMILES = smiles;
                update_model();
                update_view();
            }
        );
    }
};
const showKetcher = async function () {
    const vis = document.getElementById("ketcherModal").style.visibility;
    document.getElementById("ketcherModal").style.visibility = vis === "visible" ? "hidden" : "visible";

    const bd = document.getElementById("modalOuter");
    const bd_vis = bd.style.visibility;
    bd.style.visibility = bd_vis === "visible" ? "hidden" : "visible";

    const smi = await currentSmiles();

    if (smi) {
        document.getElementById("current-api").innerHTML = `<div class="mol">${smi}</div>`;
        draw_molecules();
    }
};

