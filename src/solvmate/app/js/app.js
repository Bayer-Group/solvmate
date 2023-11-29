
const USE_SVG = true; // Falls back to using png bitmaps if false...

// middle col: #d3f3ee
// footer: #403f4c
// header: #30638e
/*
--blue: #72A1E5;
    --indigo: #6610f2;
    --purple: #6f42c1;
    --pink: #e83e8c;
    --red: #dc3545;
    --orange: #BA5A31;
    --yellow: #ffc107;
    --green: #28a745;
    --teal: #20c997;
    --cyan: #17a2b8;
    --white: #fff;
    --gray: #797676;
    --gray-dark: #333;
    --primary: #30638E;
    --secondary: #FFA630;
    --success: #3772FF;
    --info: #C0E0DE;
    --warning: #ED6A5A;
    --danger: #ED6A5A;
    --light: #D3F3EE;
    --dark: #403F4C;

*/

var displaySmilesInElt = function (eltId, smi) {
    /*
        Displays the given smiles <smi> in the
        given DOM element with id <eltId>
    */
    fetch("SmilesInputHandler", {
        method: "POST",
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ "smiles": smi })
    }).then(res => res.json()).then(res => {
        console.log("Request complete! response:", res);
        var imgDat = res.depict_data;
        var svgDat = res.svg_data;
        var status = res.depict_status;
        console.log("about to check for success");
        if (status == "success") {
            console.log("this was a success");

            if (USE_SVG) {
                console.log("using SVG type images ...");
                var elt = document.getElementById(eltId);
                var svgElt = document.createElement("svg");
                svgElt.innerHTML = svgDat;

                // ? relative sizing of molecules?
                svgElt.setAttribute("width", "30%");
                svgElt.setAttribute("height", "30%");
                elt = elt.replaceWith(svgElt);
            }
            else {
                console.log("using PNG type images ...");
                var elt = document.getElementById(eltId);
                elt.src = "data:image/png;base64," + (imgDat);
            }
        }
        else {
            console.log("this was not passed to the success branch");
            alert("failure could not parse input!");
        }
    });
};

var displayIupacInElt = function (eltId, smi) {
    /*
    Displays the given iupac or smiles in the
    element with id eltId.
    */
    fetch("IupacInputHandler", {
        method: "POST",
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ "smiles_or_iupac": smi })
    }).then(res => res.json()).then(res => {
        console.log("Request complete! response:", res);
        var imgDat = res.depict_data;
        var svgDat = res.svg_data;
        var status = res.depict_status;
        console.log("about to check for success");
        if (status == "success") {
            console.log("this was a success");

            if (USE_SVG) {
                console.log("using SVG type images ...");
                var elt = document.getElementById(eltId);
                var svgElt = document.createElement("svg");
                svgElt.innerHTML = svgDat;

                // ? relative sizing of molecules?
                svgElt.setAttribute("width", "30%");
                svgElt.setAttribute("height", "30%");
                elt = elt.replaceWith(svgElt);
            }
            else {
                console.log("using PNG type images ...");
                var elt = document.getElementById(eltId);
                elt.src = "data:image/png;base64," + (imgDat);
            }
        }
        else {
            console.log("this was not passed to the success branch");
            alert("failure could not parse input!");
        }
    });
};


var doQuery = function (evt) {
    // Triggered whenever the user changed the
    // smiles in the smiles input widget...
    // We will then first update the view
    // within the user input so that the
    // user gets a nice visual feedback...
    var smiInput = document.getElementById("smiInput");

    var smi = smiInput.value;

    console.log("smiles = " + smi);

    // Send this smiles to the server and have it
    // send back the corresponding depiction image

    fetch("SmilesInputHandler", {
        method: "POST",
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ "smiles": smi })
    }).then(res => res.json()).then(res => {
        console.log("Request complete! response:", res);
        var imgDat = res.depict_data;
        var status = res.depict_status;
        var svgDat = res.svg_data;
        console.log("about to check for success");
        if (status == "success") {
            console.log("this was a success");
            // Be careful here, as getElementsByClassName returns an HTMLCollection
            // that will *dynamically adjust to changes*
            // We therefore convert it to an array to make sure it wont change
            // size while chaning the DOM!!!
            var pics = Array.from(document.getElementsByClassName("mol-image"));
            for (var k = 0; k < pics.length; k++) {
                console.log("iter: " + k + " / " + pics.length);
                if (USE_SVG) {
                    console.log("using SVG type images ...");
                    var elt = pics[k];
                    var svgElt = document.createElement("div");
                    svgElt.classList.add("mol-image");
                    svgElt.innerHTML = svgDat;

                    // ? relative sizing of molecules?
                    // here we could set relative sizing of images:
                    // svgElt.children[0].setAttribute("width", "30%");
                    // svgElt.children[0].setAttribute("height", "30%");

                    elt.replaceWith(svgElt);
                }
                else {
                    console.log("using PNG type images ...");
                    var elt = pics[k];
                    elt.src = "data:image/png;base64," + (imgDat);
                }
            }
        }
        else {
            console.log("this was not passed to the success branch");
            alert("failure could not parse input!");
        }
    });
};


var fetchSolventSelection = function (smi) {
    fetch("SolventSelectionFetchHandler", {
        method: "POST",
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
    }).then(res => res.json()).then(res => {
        console.log("Request complete! response:", res);

        var name_to_solvs = {};
        var selection_names = [];

        for (var q = 0; q < res["rslt"].length; q++) {
            const entry = res["rslt"][q];
            const sname = entry["selection"];
            if (selection_names.indexOf(sname) == -1) {
                selection_names.push(sname);
            }
            if (sname in name_to_solvs) {
                name_to_solvs[sname].push(entry["iupac"]);
            } else {
                name_to_solvs[sname] = [entry["iupac"]];
            }
        }

        var defaultBut = null;
        const divElt = document.getElementById("solventSelectionDiv");
        const butContainerElt = document.createElement("div");
        butContainerElt.id = "solventSelectionButsContainer";
        butContainerElt.style = "display: inline;";

        const registerButtonByName = function (sn, but) {
            console.log("registerButtonByName");
            but.classList.add("w3-button");
            but.classList.add("w3-theme");
            but.innerText = sn;
            but.onclick = function () {
                Array.from(document.getElementsByClassName("snButActive")).forEach(elt => elt.classList.remove("snButActive"));
                but.classList.add("snButActive");
                document.getElementById("textAreaSolventSelection").innerText = name_to_solvs[sn].join("\n");
            };
            butContainerElt.appendChild(but);
            console.log(butContainerElt);
            return butContainerElt;
        };

        for (var q = 0; q < selection_names.length; q++) {
            const snBut = document.createElement("button");

            if (q == 0) defaultBut = snBut;
            const sn = selection_names[q];
            registerButtonByName(sn, snBut);
        }
        const addButton = document.createElement("button");
        addButton.style = "display: inline;";
        addButton.innerText = "+";

        addButton.classList.add("w3-button");
        addButton.classList.add("w3-theme");
        addButton.onclick = function () {
            const modDialog = document.createElement("div");
            modDialog.id = "modDialog";
            modDialog.style = "position: fixed; top: 25vh; left: 25vw; width: 50vw; height: 50vh; background-color: white;";

            const pInp = document.createElement("p");
            pInp.innerText = "New Solvent Set Name:";
            const inp = document.createElement("input");
            inp.id = "modDialogNewSolventSet";
            pInp.appendChild(inp);
            modDialog.appendChild(pInp);

            const butConfirm = document.createElement("button");
            butConfirm.innerText = "OK";
            butConfirm.id = "modDialogConfirmButton";

            butConfirm.onclick = function () {
                const sn = document.getElementById("modDialogNewSolventSet").value;
                selection_names.push(sn);
                name_to_solvs[sn] = []; // Initially empty as new solvent set...

                const but = document.createElement("button");
                registerButtonByName(sn, but);
                modDialog.hidden = true;
                document.getElementById("modDialog").remove();
            };

            modDialog.appendChild(butConfirm);
            document.body.appendChild(modDialog);
        };
        divElt.appendChild(butContainerElt);
        divElt.appendChild(addButton);

        const snArea = document.createElement("div");
        snArea.id = 'textAreaSolventSelection';
        snArea.contentEditable = true;

        snArea.style = "overflow-y: auto;";

        snArea.addEventListener('input', function () {
            // When this text area changes, then it
            // means that the user is updating the
            // solvent set definitions. 
            // here we could hook up e.g. error
            // highlighting for solvent names which
            // would be important feedback from a
            // end user perspective: 
            // wrong solvent names are highlighted...
            // TODO: add highlighting logic!
            const current_sn = Array.from(document.getElementsByClassName("snButActive"))[0].innerText;
            name_to_solvs[current_sn] = Array.from(snArea.innerText.split("\n"));
        });

        const saveButton = document.createElement("button");
        saveButton.innerText = "📁";

        saveButton.classList.add("w3-button");
        saveButton.classList.add("w3-theme");

        saveButton.onclick = function () {
            var new_solvent_selections = { "iupac": [], "selection": [] };

            for (var i = 0; i < selection_names.length; i++) {
                const sn = selection_names[i];
                for (var j = 0; j < name_to_solvs[sn].length; j++) {
                    const solv = name_to_solvs[sn][j];
                    new_solvent_selections["iupac"].push(solv);
                    new_solvent_selections["selection"].push(sn);
                }
            }

            fetch("SolventSelectionStoreHandler", {
                method: "POST",
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ "new_solvent_selections": new_solvent_selections })
            }).then(res => res.json()).then(res => {
                console.log("Request complete! response:", res);

            });
        };
        divElt.appendChild(saveButton);

        divElt.appendChild(document.createElement("br"));
        divElt.appendChild(snArea);
        divElt.appendChild(document.createElement("br"));

        if (defaultBut) defaultBut.click();
    })
};



var doSolubility = function (evt) {
    // Triggered whenever the user clicks on the Run button in the solubility view.
    // We will then first launch the xTB calculation with the desired set of solvents and display as a result
    // the image with the ranked solvents.

    var resultHeaderSolub = document.getElementById("resultsHeaderSolub");
    resultHeaderSolub.innerHTML += printProgressIcon();
    var smiInput = document.getElementById("smiInput");
    var smi = smiInput.value;
    console.log("smiles = " + smi);

    var solvents = Array.from(document.getElementById("textAreaSolventSelection").innerText.split("\n"));

    fetch("SolubRecommendHandler", {
        method: "POST",
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ "smiles": smi, "solvents": solvents })
    }).then(res => res.json()).then(res => {

        doKNNSolub();

        // Cleans the progress indicator again
        resultHeaderSolub.innerHTML = "Results";
        if (!accIsVisible("resso"))
            resultHeaderSolub.click();
        console.log("Request complete! response:", res);
        var imgDat = res.depict_data;
        var status = res.depict_status;
        console.log("about to check for success");
        if (status == "success") {
            console.log("this was a success");

            var queryPic = document.getElementById("solubImage");
            queryPic.innerHTML = imgDat;
            const texts = Array.from(queryPic.getElementsByTagName("text"));
            for (var k = 0; k < texts.length; k++) {
                texts[k].onmouseover = (evt) => {
                    const popup = document.createElement("div");
                    const text = evt.target.innerHTML;
                    popup.innerHTML =
                        `<div id="plotPopUp" style='position: fixed; top: calc(50vh - 200px); left: calc(50vw - 200px); z-index:10000; width: 400px; background-color: var(--info); padding: 20px;'> 
                        <div id="plotPopUpImg"></div>
                        </div>
                        `;
                    document.body.appendChild(
                        popup
                    );
                    displayIupacInElt("plotPopUpImg", text);
                };
                texts[k].onmouseleave = function () {
                    const popUp = document.getElementById("plotPopUp");
                    if (popUp) {
                        popUp.remove();
                    }
                };
            }
        }
        else {
            console.log("this was not passed to the success branch");
            alert("failure could not parse input!");
        }
    });

};


var fetchDetails = function (eltId, smi) {
    fetch("MetaDataHandler", {
        method: "POST",
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ "smiles": smi })
    }).then(res => res.json()).then(res => {
        console.log("Request complete! response:", res);
        const elt = document.getElementById(eltId);
        elt.innerHTML = res.metadata_text;
    })
};

const showDetails = (smi) => {
    const popup = document.createElement("div");
    /* #456990 #F45B69*/
    popup.innerHTML =
        `<div id="detailsPopUp" 
        style='position: fixed; top: 25vh; left: 25vw; z-index:10000; width: 50vw; height: 50vh; background-color: var(--header-bar-bg); padding: 20px; border: 2px solid black; '> 
        <div style='padding: 10px; background-color: var(--light-gray); overflow-y: auto; height: calc(50vh - 40px);'>
            <div> <b> Recipes </b> </div>
            <div id="detailsTextDiv"> 
            <div class="my-center" id="recipesProg">
            ${printProgressIcon()}
            </div>
            </div>
            <div style="position: fixed; top: 25vh; right: 25vw;">
            <button style="text-color: #F45B69;" onclick="document.getElementById('detailsPopUp').remove();">X</button>
            </div>
        </div>
        </div>
        `;
    document.body.appendChild(
        popup
    );
    fetchDetails("detailsTextDiv", smi);
    // displayIupacInElt("plotPopUpImg", text);
};
const hideDetails = function () {
    const popUp = document.getElementById("detailsPopUp");
    if (popUp) {
        popUp.remove();
    }
};


var updateCrystPropens = function (smi) {
    fetch("CrystPropensHandler", {
        method: "POST",
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ "smiles": smi })
    }).then(res => res.json()).then(res => {
        console.log("Request complete! response:", res);
        const elt = document.getElementById("propensity");
        elt.innerHTML = res.html_resp;
    })
};

var doKNNSolub = function (evt) {
    // Triggered whenever the user clicks on the Run button in the solub view.
    // We will then request a corresponding knn search from the server side and
    // this will return the results for us to display via a REST POST request...
    console.log("call: doKNNSolub()");

    var smiInput = document.getElementById("smiInput");
    var smi = smiInput.value;
    console.log("smiles = " + smi);

    var solvents = Array.from(document.getElementById("textAreaSolventSelection").innerText.split("\n"));

    const start = solub_currentPage * solub_perPage;
    const end = (1 + solub_currentPage) * solub_perPage;

    fetch("KNNSolubRecommendHandler", {
        method: "POST",
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ "smiles": smi, "solvents": solvents, "start": start, "end": end })
    }).then(res => res.json()).then(res => {
        console.log("Request complete! response:", res);

        solub_setupPages();

        if (res["depict_status"] == "success") {
            const matches = res["matches"];

            console.log("matches.length: " + matches.length);

            const span = Math.min(end - start, matches.length);

            var print_solv_row = function (solv, conc, src) {

                return `
                <tr>
                  <td>${solv}</td>
                  <td>${conc.toFixed(2)}</td>
                  <td>${src}</td>
                </tr>
            `;
            };

            var print_tab_entry = function (match) {

                const N = match["smiles_solvents"].length;
                var similarity = match["similarity"]
                var solv_rows = "";
                for (var q = 0; q < N; q++) {
                    const solv = match["smiles_solvents"][q];
                    const src = match["source"][q];
                    const conc = match["conc"][q];

                    solv_rows += print_solv_row(solv, conc, src);
                }

                const smiles_match = match["smiles_solute"];
                const imgId = "img" + smiles_match;
                displaySmilesInElt(imgId, match["smiles_solute"]);

                const tmpl = `
          <div class="w3-row-padding w3-card w3-center w3-margin-top w3-margin-bottom" style="min-height:200px; overflow:auto">
            <br>
            <div class="w3-row-padding w3-quarter">
              <img id="${imgId}" style="position:relative;center;"
                src="/static/assets/placeholder.svg"
                ></img>
              <div style="padding: 2rem !important;">
              <p>Similarity: ${similarity}</p>
              <button onclick="showDetails('${smiles_match}');"> Details </button>
              </div>
            </div>
            <div style="padding: 10px !important; margin-left: 10px !important;">
            <table class="w3-table w3-striped w3-bordered w3-half w3-margin-bottom" style="max-height:300px; overflow:auto">
              <thead>
                <tr class="w3-theme">
                  <th>Solvent</th>
                  <th>log S</th>
                  <th>Source</th>
                </tr>
              </thead>
              <tbody>
                ${solv_rows}
              </tbody>
            </table>
            </div>
          </div>
          `
                return tmpl;
            };

            const simDiv = document.getElementById("sim_solub");
            simDiv.innerHTML = "";
            for (var q = 0; q < span; q++) {
                simDiv.innerHTML += print_tab_entry(matches[q]);
            }

        }
    }
    );

    console.log("return: doKNNSolub()");
};

var doKNNCryst = function (evt) {
    // Triggered whenever the user clicks on the Run button in the knn recommender view.
    // We will then request a corresponding knn search from the server side and
    // this will return the results for us to display via a REST POST request...
    console.log("call: doKNNCryst()");

    var smiInput = document.getElementById("smiInput");
    var smi = smiInput.value;
    console.log("smiles = " + smi);

    updateCrystPropens(smi);

    var solventSet = document.getElementsByName("solvent-set-cryst");
    var solvents = solventSet.value;

    distanceType =
        Array.from(document.getElementsByName("distance").values()).filter(rb => rb.checked)[0].value;

    var resultsHeaderCryst = document.getElementById("resultsHeaderCryst");
    resultsHeaderCryst.innerHTML += printProgressIcon();

    const start = currentPage * perPage;
    const end = (1 + currentPage) * perPage;

    var rec_type;
    if (distanceType === "ecfp")
        rec_type = "ecfp";
    else
        rec_type = "2DDescriptors";

    console.assert(rec_type == "2DDescriptors" || rec_type == "ecfp");
    fetch("KNNRecommendHandler", {
        method: "POST",
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ "smiles": smi, "solvents": solvents, "start": start, "end": end, "rec_type": rec_type })
    }).then(res => res.json()).then(res => {
        console.log("Request complete! response:", res);

        setupPages();

        // Cleans the progress indicator again
        resultsHeaderCryst.innerHTML = "Results";
        if (!accIsVisible("rescr"))
            resultsHeaderCryst.click();

        if (res["depict_status"] == "success") {
            const matches = res["matches"];

            console.log("matches.length: " + matches.length);

            const span = Math.min(end - start, matches.length);

            // At this point in time, we got the results of the KNN search returned
            // within the json of the `res` object.

            // This object contains the following columns:
            // (example taken from public domain nova data)
            /*
                distance: 
                7.745966692414834
                similarity: 
                0.11433841851550565
                smiles_solute: 
                "CCCC(=O)Nc1ccc(Cl)c(Cl)c1"
                smiles_solvents: 
                ['ClC(Cl)Cl', 'Cc1ccccc1', 'CC#N']
                source:
                [ESS,NOVA,NOVA]
                ...
    
            */


            var print_solv_row = function (solv, src) {

                return `
                <tr>
                  <td>${solv}</td>
                  <td>${src}</td>
                </tr>
            `;
            };

            var print_tab_entry = function (match) {

                const N = match["smiles_solvents"].length;
                var similarity = match["similarity"]
                var solv_rows = "";
                for (var q = 0; q < N; q++) {
                    const solv = match["smiles_solvents"][q];
                    const src = match["source"][q];

                    solv_rows += print_solv_row(solv, src);
                }

                const smiles_match = match["smiles_solute"];
                const imgId = "img" + smiles_match;
                displaySmilesInElt(imgId, match["smiles_solute"]);

                const tmpl = `
          <div class="w3-row-padding w3-card w3-center w3-margin-top w3-margin-bottom" style="min-height:200px; overflow:auto">
            <br>
            <div class="w3-row-padding w3-quarter">
              <img id="${imgId}" style="position:relative;center;"
                src="/static/assets/placeholder.svg"
                ></img>
              <div style="padding: 2rem !important;">
              <p>Similarity: ${similarity}</p>
              <button onclick="showDetails('${smiles_match}');"> Details </button>
              </div>
            </div>
            <div style="padding: 10px !important; margin-left: 10px !important;">
            <table class="w3-table w3-striped w3-bordered w3-half w3-margin-bottom" style="max-height:300px; overflow:auto">
              <thead>
                <tr class="w3-theme">
                  <th>Solvent</th>
                  <th>Source</th>
                </tr>
              </thead>
              <tbody>
                ${solv_rows}
              </tbody>
            </table>
            </div>
          </div>
          `
                return tmpl;
            };

            const simDiv = document.getElementById("sim");
            simDiv.innerHTML = "";
            for (var q = 0; q < span; q++) {
                simDiv.innerHTML += print_tab_entry(matches[q]);
            }

        }
    }
    );

    console.log("return: doKNNCryst()");
};

var printProgressIcon = function () {

    return `
    <div >
<svg
   style="animation: spin 1s infinite linear;"
   width="4.512516mm"
   height="4.512516mm"
   viewBox="0 0 26.512516 26.512516"
   version="1.1"
   id="svg5"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:svg="http://www.w3.org/2000/svg">
  <defs
     id="defs2" />
  <g
     id="progLayer1"
     transform="translate(-52.319189,-67.189848)">
    <path
       id="path879"
       style="fill:none;stroke:#f45b69;stroke-width:4;stroke-miterlimit:4;stroke-dasharray:none;stroke-opacity:1"
       d="m 76.831705,80.446106 c 0,6.21666 -5.039598,11.256258 -11.256258,11.256258 -6.21666,0 -11.256258,-5.039598 -11.256258,-11.256258 0,-6.21666 5.039598,-11.256258 11.256258,-11.256258 1.774793,0.18183 3.397489,0.627657 4.824593,1.300765" />
  </g>
</svg>
</div>
`;


};

// Paging for the crystal conditions search
perPage = 20;
startPage = 0;
endPage = 1000;
currentPage = 0;
var setupPages = function () {
    const pDiv = document.getElementById("pager");
    pDiv.innerHTML = "";

    for (var p = currentPage - 5; p < currentPage + 6; p++) {
        if (p < startPage || p >= endPage) continue;

        if (p == currentPage)
            pDiv.innerHTML += `<b style="color: red;" > ${p} </b>`;
        else
            pDiv.innerHTML += `<b onclick="currentPage=${p}; doKNNCryst(); document.getElementById('sim').scrollIntoView();" style="cursor: pointer;"> ${p} </b>`;
    }

};

// Paging for the solub search
solub_perPage = 20;
solub_startPage = 0;
solub_endPage = 1000;
solub_currentPage = 0;
var solub_setupPages = function () {
    const pDiv = document.getElementById("pager_solub");
    pDiv.innerHTML = "";

    for (var p = solub_currentPage - 5; p < solub_currentPage + 6; p++) {
        if (p < solub_startPage || p >= solub_endPage) continue;

        if (p == solub_currentPage)
            pDiv.innerHTML += `<b style="color: red;" > ${p} </b>`;
        else
            pDiv.innerHTML += `<b onclick="solub_currentPage=${p}; doKNNSolub(); document.getElementById('sim_solub').scrollIntoView();" style="cursor: pointer;"> ${p} </b>`;
    }

};

const getMolBlockFromEditor = function (molEditor,) {
    molPaintJS.dump(molEditor, "printOutput", "V3000");
    return document.getElementById("printOutput").innerHTML;
}

const setupMolEditor = function () {

    mp = molPaintJS.newContext("molEditor", { sizeX: 400, sizeY: 400, debugId: "molDebug" });
    mp.init();
    mp.setChangeListener(function () {
        // If this here is triggered it means that the user
        // has changed the molecule within the mol editor.
        // Therefore, we need to make sure that all of the
        // other site elements are updated accordingly.
        console.log("mp.changeListener()");

        const smiInput = document.getElementById("smiInput");

        const mdlMol = getMolBlockFromEditor("molEditor");
        fetch("MDLMolToSmilesHandler", {
            method: "POST",
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ "mdl_mol": mdlMol })
        }).then(res => res.json()).then(res => {
            console.log("Request complete! response:", res);
            smiInput.value = res["smiles"];
            doQuery();
        });
    });
};

window.addEventListener('load', function () {
    console.log("web page loaded!");
    var appDiv = document.createElement("div");
    appDiv.classList.add("appDiv");

    var smiInput = document.getElementById("smiInput");
    smiInput.id = "smiInput";

    var runSol = document.getElementById("runSol");

    // Once the user clicks onto the smiles input widget
    // then we want to update the app accordingly!
    // So we need to hook the event listener onto
    // change events within the input element:
    smiInput.addEventListener(
        "change", doQuery
    );

    runSol.addEventListener(
        "click", doSolubility
    );

    var runCrys = document.getElementById("runCrys");


    runCrys.addEventListener(
        "click", function () { currentPage = 0; doKNNCryst(); }
    );

    fetchSolventSelection();

    setupMolEditor();
}
);

