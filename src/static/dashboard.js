// Dashboard JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Class Distribution Chart
    const classCtx = document.getElementById('classChart').getContext('2d');
    new Chart(classCtx, {
        type: 'doughnut',
        data: {
            labels: ['Normal Transactions', 'Fraud Transactions'],
            datasets: [{
                data: [284315, 492],
                backgroundColor: ['#4CAF50', '#F44336'],
                borderWidth: 2,
                borderColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        usePointStyle: true
                    }
                }
            }
        }
    });

    // Amount Distribution Chart (simplified histogram)
    const amountCtx = document.getElementById('amountChart').getContext('2d');
    new Chart(amountCtx, {
        type: 'bar',
        data: {
            labels: ['$0-50', '$50-100', '$100-200', '$200-500', '$500+'],
            datasets: [{
                label: 'Transaction Count',
                data: [150000, 80000, 40000, 12000, 2807],
                backgroundColor: 'rgba(54, 162, 235, 0.6)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return (value / 1000).toFixed(0) + 'k';
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });

    // Add some interactivity to metric cards
    const metricCards = document.querySelectorAll('.metric-card');
    metricCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-2px)';
            this.style.boxShadow = '0 8px 25px rgba(0,0,0,0.15)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
            this.style.boxShadow = '0 4px 12px rgba(0,0,0,0.1)';
        });
    });

    // Add animation to feature items
    const featureItems = document.querySelectorAll('.feature-item');
    featureItems.forEach((item, index) => {
        item.style.animationDelay = `${index * 0.1}s`;
        item.classList.add('animate-in');
    });

    // Add click handlers for file items
    const fileItems = document.querySelectorAll('.file-item');
    fileItems.forEach(item => {
        item.addEventListener('click', function() {
            this.style.backgroundColor = '#e3f2fd';
            setTimeout(() => {
                this.style.backgroundColor = '';
            }, 200);
        });
    });
}); 