// Wildfire Nexus Map - Leaflet Integration

let map;
let markersLayer;
let refreshInterval;

const API_BASE = '/api/v1';
const COLORS = {
    critical: '#dc3545',
    warning: '#fd7e14',
    info: '#17a2b8'
};

// Initialize map
function initMap() {
    // Center on Siberia (configurable)
    map = L.map('map').setView([60, 95], 4);
    
    // Add OpenStreetMap tiles
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);
    
    markersLayer = L.layerGroup().addTo(map);
    
    // Load initial data
    loadFires();
    
    // Auto-refresh every 30 seconds
    refreshInterval = setInterval(loadFires, 30000);
    
    // Hide loading indicator
    document.getElementById('loading').style.display = 'none';
}

// Load fire events from API
async function loadFires() {
    try {
        const response = await fetch(`${API_BASE}/events?status=active&limit=500`);
        if (!response.ok) throw new Error('API error');
        
        const events = await response.json();
        updateMarkers(events);
        updateStats(events);
        
        // Update timestamp
        const now = new Date();
        document.getElementById('last-update').textContent = now.toLocaleTimeString();
        
    } catch (error) {
        console.error('Failed to load fires:', error);
    }
}

// Update map markers
function updateMarkers(events) {
    markersLayer.clearLayers();
    
    events.forEach(event => {
        const color = COLORS[event.risk_level] || COLORS.info;
        
        // Create marker with custom icon
        const marker = L.circleMarker(
            [event.centroid_lat, event.centroid_lon],
            {
                radius: getMarkerRadius(event),
                fillColor: color,
                color: '#fff',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.7
            }
        );
        
        // Add popup
        marker.bindPopup(createPopupContent(event));
        
        markersLayer.addLayer(marker);
    });
}

// Get marker size based on severity
function getMarkerRadius(event) {
    const baseRadius = 8;
    const sizeByRisk = {
        critical: 15,
        warning: 12,
        info: baseRadius
    };
    return sizeByRisk[event.risk_level] || baseRadius;
}

// Create popup HTML
function createPopupContent(event) {
    const riskLabel = event.risk_level.charAt(0).toUpperCase() + event.risk_level.slice(1);
    
    let html = `
        <div style="min-width: 200px;">
            <h3 style="margin: 0 0 10px 0; color: ${COLORS[event.risk_level]}">
                🔥 Fire #${event.id.substring(0, 6)}
            </h3>
            <p><strong>Status:</strong> ${event.status}</p>
            <p><strong>Risk:</strong> ${riskLabel}</p>
            <p><strong>Points:</strong> ${event.point_count}</p>
            <p><strong>Area:</strong> ~${event.area_estimate_ha.toFixed(0)} ha</p>
    `;
    
    if (event.max_frp > 0) {
        html += `<p><strong>Max FRP:</strong> ${event.max_frp.toFixed(1)} MW</p>`;
    }
    
    if (event.nearest_settlement) {
        html += `<p><strong>Near:</strong> ${event.nearest_settlement}</p>`;
    }
    
    html += `
            <p style="font-size: 12px; color: #666;">
                Last seen: ${new Date(event.last_seen).toLocaleString()}
            </p>
            <a href="/api/v1/events/${event.id}" target="_blank" style="font-size: 12px;">
                View details →
            </a>
        </div>
    `;
    
    return html;
}

// Update statistics panel
function updateStats(events) {
    const stats = {
        active: events.length,
        critical: events.filter(e => e.risk_level === 'critical').length,
        warning: events.filter(e => e.risk_level === 'warning').length
    };
    
    document.getElementById('active-count').textContent = stats.active;
    document.getElementById('critical-count').textContent = stats.critical;
    document.getElementById('warning-count').textContent = stats.warning;
}

// Handle URL hash for direct event linking
function handleHash() {
    const hash = window.location.hash;
    if (hash.startsWith('#event=')) {
        const eventId = hash.substring(7);
        // Could auto-focus on this event
        console.log('Direct link to event:', eventId);
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initMap();
    handleHash();
});

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    if (refreshInterval) {
        clearInterval(refreshInterval);
    }
});
