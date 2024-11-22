document.addEventListener('DOMContentLoaded', function() {
    const servicesSec = document.getElementById('Service');
    const modelDropdown = document.getElementById('ModelDropdown');
    const bodyWeb = document.querySelector('.BodyWeb'); // Add this line

    if (!servicesSec || !modelDropdown || !bodyWeb) {
        console.error('Required elements not found');
        return;
    }

    servicesSec.addEventListener('click', function(e) {
        e.stopPropagation();
        modelDropdown.classList.toggle('show');
        bodyWeb.classList.toggle('shifted'); // Add this line
    });
 
    document.addEventListener('click', function(e) {
        if (!servicesSec.contains(e.target)) {
            modelDropdown.classList.remove('show');
            bodyWeb.classList.remove('shifted'); // Add this line
        }
    });

    // Log initial state
});