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
            $('#fact-category').text(result.label);
            console.log(result)
        });
    }


    bindEvents();

}($))