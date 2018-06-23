(function ($) {
    var BACKEND = 'http://0.0.0.0:8005';

    function bindEvents() {
        var $button = $('#check-fact-btn');

        $button.click(function () {
            var $input = $('#check-fact-input');
            checkFact($input.val());
        });
    }

    function getCategory(fact) {
        return $.ajax({
            type: 'GET',
            url: BACKEND + '/category?q=' + fact,
            dataType: "json"
        });
    }

    function checkFact(fact) {
        getCategory(fact).done(function (result) {
            $('#fact-category').text(result.pred.label + ' &score=' + result.pred.score);
            renderAtt(result);

        });
    }


    bindEvents();

    function renderAtt(result) {
        var att = result.att;

        var $viewer = $('#viewer');

        var $p = $('<p>');

        var orig_text = result.text;
        var result_html = orig_text;
        for (var i = 0; i < att['text'].length; i++) {
            var regex = new RegExp('\\b' + att['text'][i] + '\\b', 'i');
            var index = orig_text.search(regex);
            if (index !== -1) {
                result_html = result_html.replace(regex, '<span style="background-color:' + 'hsl(151, 100%,' + (1 - (att['att_score'][i] * Math.pow(100, att['att_score'][i]))) * 100 + '%)' + '">' + orig_text.substring(index, index + att['text'][i].length) + '</span>');
            }
        }
        $p.html(result_html);
        $p.appendTo($viewer);

    }

}($));