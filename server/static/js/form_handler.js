const MAX_CHAR_NUM = 200;
const LIGHT_NEITHER_CLASS = ".led-neither";
const LIGHT_HATE_CLASS = ".led-hate";
const LIGHT_OFFENSIVE_CLASS = ".led-offensive";
const CONFIDENCE_NEITHER_ID = "#neither-value";
const CONFIDENCE_HATE_ID = "#hate-value";
const CONFIDENCE_OFFENSIVE_ID = "#offensive-value";

function displayLoadingSpinner(){
    $('#loading').show()
}

function hideLoadingSpinner(){
    $('#loading').hide()
}

function setLightOff(light_class, confidence_elem_id){
    $(light_class).css("background-color", "#DCDCDC");
    $(light_class).css("box-shadow", "rgba(0, 0, 0, 0.2) 0 -1px 7px 1px, inset #A9A9A9 0 -1px 9px, #D3D3D3 0 2px 12px");
    $(confidence_elem_id).text('')
}

function setLightOn(light_class, confidence_elem_id, confidence){
    $(light_class).css("background-color", "#F00");
    $(light_class).css("box-shadow", "rgba(0, 0, 0, 0.2) 0 -1px 7px 1px, inset #441313 0 -1px 9px, rgba(255, 0, 0, 0.5) 0 2px 12px");
    var confidence_in_percentage = confidence * 100;
    var rounded_confidence = Math.round(confidence_in_percentage * 10) / 10;
    $(confidence_elem_id).text("(" + rounded_confidence + "%)")
}

function updateLight(label, confidence) {
    switch (label) {
        case null:
            setLightOff(LIGHT_NEITHER_CLASS, CONFIDENCE_NEITHER_ID);
            setLightOff(LIGHT_HATE_CLASS, CONFIDENCE_HATE_ID);
            setLightOff(LIGHT_OFFENSIVE_CLASS, CONFIDENCE_OFFENSIVE_ID);
            break;
        case "hate":
            setLightOff(LIGHT_NEITHER_CLASS, CONFIDENCE_NEITHER_ID);
            setLightOn(LIGHT_HATE_CLASS, CONFIDENCE_HATE_ID, confidence);
            setLightOff(LIGHT_OFFENSIVE_CLASS, CONFIDENCE_OFFENSIVE_ID);
            break;
        case "offensive":
            setLightOff(LIGHT_NEITHER_CLASS, CONFIDENCE_NEITHER_ID);
            setLightOff(LIGHT_HATE_CLASS, CONFIDENCE_HATE_ID);
            setLightOn(LIGHT_OFFENSIVE_CLASS, CONFIDENCE_OFFENSIVE_ID, confidence);
            break;
        case "neither":
            setLightOn(LIGHT_NEITHER_CLASS, CONFIDENCE_NEITHER_ID, confidence);
            setLightOff(LIGHT_HATE_CLASS, CONFIDENCE_HATE_ID);
            setLightOff(LIGHT_OFFENSIVE_CLASS, CONFIDENCE_OFFENSIVE_ID);
            break;
        default:
            setLightOff(LIGHT_NEITHER_CLASS, CONFIDENCE_NEITHER_ID);
            setLightOff(LIGHT_HATE_CLASS, CONFIDENCE_HATE_ID);
            setLightOff(LIGHT_OFFENSIVE_CLASS, CONFIDENCE_OFFENSIVE_ID);
    }
}

function submitClassificationRequest() {
    var validated = true; // TODO we might want to perform some validation against the input, e.g. length > 0
    var input_text = $('#text').val().trim();
    console.log(input_text);

    if (validated){
        displayLoadingSpinner();
        $.ajax({
            url: "./classify",
            type: 'post',
            dataType: "json",
            data: {
                "input_text": input_text
            },
            success: function (response) {
                hideLoadingSpinner();
                updateLight(response.predicted_label, response.confidence);
            },
            error: function () {
                hideLoadingSpinner();
                updateLight(null, 0);
                swal("Oops...", "Something went wrong!", "error");
            }
        });
    }

    return false;
}

$(document).ready(function () {
    $('#text').val('')
    $('#text').keyup(function () {
        var text_length = $('#text').val().length;
        if (text_length >= MAX_CHAR_NUM) {
            $('#count_message').text(' you have reached ' + MAX_CHAR_NUM + ' char limit');
        } else {
            var char = MAX_CHAR_NUM - text_length;
            $('#count_message').text(char + ' characters left');
        }
    });

    // light bulbs related
    // var $winHeight = $( window ).height()
    // $( '.container' ).height( $winHeight );
});
