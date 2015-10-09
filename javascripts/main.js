console.log('This would be the main JS file.');

$(document).ready(function() {
    var s = $("#sticker");
    $(window).scroll(function() {
        var windowpos = $(window).scrollTop();
        if (windowpos >= 272.39) {
            s.addClass("stick");
        } else {
            s.removeClass("stick");
        }
    });
});