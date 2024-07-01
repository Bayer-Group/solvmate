/*
    A simple javascript implementation of a solvent chooser.
    Design goals:
    -------------
    - Persistence within frontend via cookies,
      thus keeping the backend stateless
    - Users of Solvmate are probably powerusers 
      and have specific solvent sets that will
      always be of interest to them
      -> they should not have to retype them!
    - Therefore, this module implements persistence
      within the client. 

    How it works:
    --------------
    The user can choose between solvent presets 1-5.
    Each button represents one preset.
    Furthermore, the contents of a preset can be freely
    altered via the text area.

    Users can load both SMILES, IUPAC names, and common
    solvent abbreviations (e.g. THF, DMSO, ...) 
    for convenience in the general case, and broad applicability
    in case of special solvents.
*/

const showSolventsDialog = function () {
    loadSolventsDialog();
    const vis = document.getElementById("solventsDialogModal").style.visibility;
    document.getElementById("solventsDialogModal").style.visibility = vis === "visible" ? "hidden" : "visible";
};

function setCookie(name, value, days) {
    let expires = "";
    if (days) {
        const date = new Date();
        date.setTime(date.getTime() + (days * 24 * 60 * 60 * 1000));
        expires = "; expires=" + date.toUTCString();
    }
    document.cookie = name + "=" + (value || "") + expires + "; path=/";
}

function getCookie(name) {
    const nameEQ = name + "=";
    const ca = document.cookie.split(';');
    for (let i = 0; i < ca.length; i++) {
        let c = ca[i];
        while (c.charAt(0) === ' ') c = c.substring(1, c.length);
        if (c.indexOf(nameEQ) === 0) return c.substring(nameEQ.length, c.length);
    }
    return null;
}

function eraseCookie(name) {
    document.cookie = name + '=; Max-Age=-99999999;';
}

const DEFAULT_SOLVENT_SETTINGS = [
    ["CCCO", "CCO", "CO", "O"].join("|"),
    ["CCCN", "CCN", "CN", "N"].join("|"),
    ["CCCP", "CCP", "CP", "P"].join("|"),
    ["CCCF", "CCF", "CF", "F"].join("|"),
    ["CCCC", "CCC", "CC", "C"].join("|"),
];

const _sp_prefix = "solvent-set-";

var _sp_active = getCookie("last-sp-active");
if (_sp_active)
    _sp_active = Number(_sp_active);
else
    _sp_active = 0;

const loadSolventsDialog = function () {
    const elt = document.getElementById("solvent-settings");
    elt.innerHTML = "";
    var buttonDiv = document.createElement('div');
    buttonDiv.style.display = "bloack";
    for (var idx = 0; idx < 5; idx++) {
        var button = document.createElement('button');
        button.textContent = idx;
        button.my_index = idx;
        button.addEventListener('click', function (event) {
            const button = event.target;
            _sp_active = button.my_index;
            loadSolventsDialog();
        });
        if (idx == _sp_active) {
            button.classList.add("primary");
        } else {
            button.classList.add("secondary");
        }
        buttonDiv.appendChild(button);
    }
    elt.appendChild(buttonDiv);

    const cookie_key = _sp_prefix + _sp_active;
    var cookie_val = getCookie(cookie_key);
    if (!cookie_val) {
        // initialize with defaults.
        cookie_val = DEFAULT_SOLVENT_SETTINGS[_sp_active];
    }
    var textArea = document.createElement('textarea');
    textArea.id = 'ta-' + cookie_key;
    textArea.rows = 10;
    textArea.cols = 30;
    textArea.value = cookie_val.split("|").join("\n");
    elt.appendChild(textArea);

    setCookie('last-sp-active', _sp_active, 1000);
};

const commitSolventsDialog = function () {
};