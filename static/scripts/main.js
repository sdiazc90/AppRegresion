$(document).ready(function () {
    // 1) Al cargar, dejamos todo deshabilitado menos Marca
    $('#modelo, #motor, #transmision, #traccion').prop('disabled', true);

    // 2) Cuando se selecciona una marca
   // Cuando cambia la marca
$('#marca').change(function () {
    const marcaSeleccionada = $(this).val();
    console.log('Marca seleccionada:', marcaSeleccionada);

    // Limpiar selects sin añadir opción "Seleccione..."
    $('#modelo, #motor, #transmision, #traccion').empty().prop('disabled', true);

    if (marcaSeleccionada) {
        $.post('/get_modelos', { marca: marcaSeleccionada.toLowerCase() }, function (modelos) {
            console.log('Modelos recibidos:', modelos);
            modelos.forEach(m => {
                const display = m.charAt(0).toUpperCase() + m.slice(1).toLowerCase();
                $('#modelo').append(`<option value="${m}">${display}</option>`);
            });
            $('#modelo').prop('disabled', false);
        });
    }
});


    // 3) Cuando se selecciona un modelo
    $('#modelo').change(function () {
        const modeloSeleccionado = $(this).val();
        console.log('Modelo seleccionado:', modeloSeleccionado);

        // Reseteo de los dropdowns dependientes
        // Reseteo de los dropdowns dependientes
$('#motor, #transmision, #traccion').empty().prop('disabled', true);

        if (modeloSeleccionado) {
            // Motores
            $.post('/get_motores', { modelo: modeloSeleccionado }, function (motores) {
                console.log('Motores recibidos:', motores);
                motores.forEach(m => {
                    $('#motor').append(`<option value="${m}">${m}</option>`);
                });
                $('#motor').prop('disabled', false);
            });

            // Transmisiones
           $.post('/get_transmisiones', { modelo: modeloSeleccionado }, function (transmisiones) {
    console.log('Transmisiones recibidas:', transmisiones);
    transmisiones.forEach(t => {
        const display = t.charAt(0).toUpperCase() + t.slice(1).toLowerCase();
        $('#transmision').append(`<option value="${t}">${display}</option>`);
    });
    $('#transmision').prop('disabled', false);
});


            // Tracciones
            $.post('/get_tracciones', { modelo: modeloSeleccionado }, function (tracciones) {
    console.log('Tracciones recibidas:', tracciones);
    tracciones.forEach(t => {
        const display = t.charAt(0).toUpperCase() + t.slice(1).toLowerCase();
        $('#traccion').append(`<option value="${t}">${display}</option>`);
    });
    $('#traccion').prop('disabled', false);
});

        }
    });
});

