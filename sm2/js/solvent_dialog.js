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

    const bd = document.getElementById("modalOuter");
    const bd_vis = bd.style.visibility;
    bd.style.visibility = bd_vis === "visible" ? "hidden" : "visible";
};

const _USE_COOKIES = false;
function setCookie(name, value, days) {

    if (_USE_COOKIES) {
        let expires = "";
        if (days) {
            const date = new Date();
            date.setTime(date.getTime() + (days * 24 * 60 * 60 * 1000));
            expires = "; expires=" + date.toUTCString();
        }
        document.cookie = name + "=" + (value || "") + expires + "; path=/";
    }
    else {
        localStorage.setItem(name, value);
    }
}

function getCookie(name) {

    if (_USE_COOKIES) {
        const nameEQ = name + "=";
        const ca = document.cookie.split(';');
        for (let i = 0; i < ca.length; i++) {
            let c = ca[i];
            while (c.charAt(0) === ' ') c = c.substring(1, c.length);
            if (c.indexOf(nameEQ) === 0) return c.substring(nameEQ.length, c.length);
        }
        return null;
    } else {
        return localStorage.getItem(name);
    }
}

function eraseCookie(name) {
    document.cookie = name + '=; Max-Age=-99999999;';
}


const _sp_prefix = "solvent-set-";

var _sp_active = getCookie("last-sp-active");
if (_sp_active)
    _sp_active = Number(_sp_active);
else
    _sp_active = 0;


const _nameForSlot = function (idx) {
    const nme = getCookie("name-" + idx);
    if (!nme)
        return DEFAULT_SOLVENT_SETTINGS_NAMES[idx];
    else
        return nme;
};

const loadSolventsDialog = function () {
    const elt = document.getElementById("solvent-settings");
    elt.innerHTML = "";
    var buttonDiv = document.createElement('div');
    buttonDiv.style.display = "block";
    for (var idx = 0; idx < 5; idx++) {
        var button = document.createElement('button');
        button.textContent = _nameForSlot(idx);
        button.my_index = idx;
        button.style.fontSize = "14px";
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
    textArea.id = 'textarea-solvent-setting';
    textArea.rows = 10;
    textArea.cols = 35;
    textArea.value = cookie_val.split("|").join("\n");
    elt.appendChild(textArea);

    var nameAreaLabel = document.createElement("p");
    nameAreaLabel.style.marginTop = "30px";
    nameAreaLabel.innerText = "name:";
    elt.appendChild(nameAreaLabel);

    var nameArea = document.createElement('textarea');
    nameArea.id = 'textarea-solvent-setting-name';
    nameArea.rows = 1;
    nameArea.cols = 10;
    nameArea.value = _nameForSlot(_sp_active);
    elt.appendChild(nameArea);
    setCookie('last-sp-active', _sp_active, 1000);
};

const commitSolventsDialog = function () {
    const ta = document.getElementById('textarea-solvent-setting');
    const tan = document.getElementById('textarea-solvent-setting-name');
    setCookie(_sp_prefix + _sp_active, ta.value.split("\n").join("|"), 1000);
    setCookie("name-" + _sp_active, tan.value.trim());
    showSolventsDialog();
};


const get_selected_solvents = function () {
    return Array.from(getCookie(_sp_prefix + _sp_active).split("|"));
};