import * as React from "react";
import { motion } from "motion/react";
import {
  MapPin,
  Camera,
  Save,
  Search,
  ArrowRight,
  Crosshair,
  RotateCcw,
  AlertCircle,
  CheckCircle2,
  ExternalLink,
} from "lucide-react";
import { setOptions, importLibrary } from "@googlemaps/js-api-loader";
import { PageHeader, PageContainer } from "@/components/common/page-header";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Separator } from "@/components/ui/separator";
import { camerasApi } from "@/services/api";
import { useApi } from "@/hooks/use-api";
import type { CameraWithGraph, CameraMapsConfig } from "@/types/api";
import { cn } from "@/lib/utils";

// Default fallback coordinates (Chandigarh center) if camera coordinates are not set
const DEFAULT_CENTER = { lat: 30.7333, lng: 76.7794 };

export default function CameraLocations() {
  const { data: cameras, loading, refetch } = useApi<CameraWithGraph[]>(() => camerasApi.list());
  const { data: mapsConfig, loading: mapsConfigLoading } = useApi<CameraMapsConfig>(() =>
    camerasApi.mapsConfig(),
  );

  const [selectedId, setSelectedId] = React.useState<string | null>(null);
  const [search, setSearch] = React.useState("");

  // Select first camera automatically once loaded if none selected
  React.useEffect(() => {
    if (!selectedId && cameras && cameras.length > 0) {
      setSelectedId(cameras[0].id);
    }
  }, [cameras, selectedId]);

  const filtered =
    cameras?.filter(
      (c) =>
        c.id.toLowerCase().includes(search.toLowerCase()) ||
        c.location?.toLowerCase().includes(search.toLowerCase()),
    ) ?? [];

  const selected = cameras?.find((c) => c.id === selectedId) ?? null;

  return (
    <PageContainer>
      <PageHeader
        title="Camera Locations"
        description="Camera network management, Google Maps geolocation, and topology configuration"
      />

      <div className="grid gap-5 lg:grid-cols-[340px_1fr]">
        {/* Camera list */}
        <div className="space-y-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-faint" />
            <Input
              placeholder="Search cameras by ID or location…"
              className="pl-9"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>

          <div className="space-y-2">
            {loading ? (
              [1, 2, 3].map((i) => <Skeleton key={i} className="h-20 w-full rounded-lg" />)
            ) : filtered.length === 0 ? (
              <div className="rounded-lg border border-dashed border-border py-8 text-center text-sm text-ink-muted">
                No cameras found matching &ldquo;{search}&rdquo;.
              </div>
            ) : (
              filtered.map((cam) => (
                <CameraListItem
                  key={cam.id}
                  camera={cam}
                  isSelected={selectedId === cam.id}
                  onSelect={() => setSelectedId(cam.id)}
                />
              ))
            )}
          </div>
        </div>

        {/* Detail panel */}
        {selected ? (
          <CameraDetail
            key={selected.id}
            camera={selected}
            allCameras={cameras || []}
            mapsConfig={mapsConfig || null}
            mapsConfigLoading={mapsConfigLoading}
            onSelectCamera={(id) => setSelectedId(id)}
            onUpdate={refetch}
          />
        ) : (
          <Card>
            <CardContent className="flex h-[450px] flex-col items-center justify-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-surface-muted">
                <MapPin className="h-6 w-6 text-ink-faint" />
              </div>
              <p className="text-sm text-ink-muted">Select a camera from the list to view and edit details</p>
            </CardContent>
          </Card>
        )}
      </div>
    </PageContainer>
  );
}

function CameraListItem({
  camera,
  isSelected,
  onSelect,
}: {
  camera: CameraWithGraph;
  isSelected: boolean;
  onSelect: () => void;
}) {
  const hasGps = camera.latitude != null && camera.longitude != null;

  return (
    <motion.div
      initial={{ opacity: 0, x: 8 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.15 }}
      className={cn(
        "cursor-pointer rounded-lg border p-3.5 transition-all",
        isSelected
          ? "border-primary bg-primary/5 shadow-sm ring-1 ring-primary/20"
          : "border-border bg-surface hover:border-border-strong hover:bg-surface-muted/40",
      )}
      onClick={onSelect}
    >
      <div className="flex items-start gap-3">
        <div
          className={cn(
            "flex h-9 w-9 shrink-0 items-center justify-center rounded-md",
            isSelected ? "bg-primary text-white" : "bg-surface-muted text-ink-secondary",
          )}
        >
          <Camera className="h-4 w-4" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-1">
            <span className="text-sm font-semibold text-ink">{camera.id}</span>
            <Badge variant={camera.is_active ? "success" : "secondary"} className="text-[10px]">
              {camera.is_active ? "Active" : "Inactive"}
            </Badge>
          </div>
          <div className="truncate text-xs font-medium text-ink-muted mt-0.5">{camera.location}</div>
          <div className="mt-1.5 flex items-center justify-between text-[11px]">
            {hasGps ? (
              <span className="inline-flex items-center gap-1 font-mono text-[10px] text-emerald-600 dark:text-emerald-400">
                <MapPin className="h-3 w-3" />
                {camera.latitude?.toFixed(4)}, {camera.longitude?.toFixed(4)}
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 text-[10px] text-ink-faint">
                <MapPin className="h-3 w-3" />
                No GPS set
              </span>
            )}
            <span className="text-ink-muted">
              {camera.connects_to.length} {camera.connects_to.length === 1 ? "link" : "links"}
            </span>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

function CameraDetail({
  camera,
  allCameras,
  mapsConfig,
  mapsConfigLoading,
  onSelectCamera,
  onUpdate,
}: {
  camera: CameraWithGraph;
  allCameras: CameraWithGraph[];
  mapsConfig: CameraMapsConfig | null;
  mapsConfigLoading: boolean;
  onSelectCamera: (id: string) => void;
  onUpdate: () => void;
}) {
  const [location, setLocation] = React.useState(camera.location);
  const [startTime, setStartTime] = React.useState(camera.start_time || "");
  const [latitude, setLatitude] = React.useState<number | null>(camera.latitude ?? null);
  const [longitude, setLongitude] = React.useState<number | null>(camera.longitude ?? null);
  const [googlePlaceId, setGooglePlaceId] = React.useState<string | null>(
    camera.google_place_id ?? null,
  );
  const [transitEdits, setTransitEdits] = React.useState<Record<string, number>>({});
  const [saving, setSaving] = React.useState(false);
  const [statusMsg, setStatusMsg] = React.useState<{ text: string; type: "success" | "error" } | null>(
    null,
  );

  // Sync state if camera changes
  React.useEffect(() => {
    setLocation(camera.location);
    setStartTime(camera.start_time || "");
    setLatitude(camera.latitude ?? null);
    setLongitude(camera.longitude ?? null);
    setGooglePlaceId(camera.google_place_id ?? null);
    setTransitEdits({});
    setStatusMsg(null);
  }, [camera]);

  const handleCoordinatesChange = (
    lat: number,
    lng: number,
    placeId?: string | null,
    placeName?: string,
  ) => {
    setLatitude(lat);
    setLongitude(lng);
    if (placeId !== undefined) {
      setGooglePlaceId(placeId);
    }
    if (placeName && (!location || location === camera.location)) {
      setLocation(placeName);
    }
  };

  const handleResetLocation = () => {
    setLatitude(camera.latitude ?? null);
    setLongitude(camera.longitude ?? null);
    setGooglePlaceId(camera.google_place_id ?? null);
    setLocation(camera.location);
  };

  const handleSave = async () => {
    setSaving(true);
    setStatusMsg(null);
    try {
      // 1. Update location & coordinates if latitude and longitude are specified
      if (latitude !== null && longitude !== null) {
        await camerasApi.updateLocation(camera.id, {
          latitude: Number(latitude),
          longitude: Number(longitude),
          google_place_id: googlePlaceId?.trim() || undefined,
          location: location.trim() || undefined,
        });
      } else if (location.trim() !== camera.location) {
        await camerasApi.patch(camera.id, {
          location: location.trim(),
        });
      }

      // 2. Update start_time or transit_updates via PATCH
      const transitUpdates: Record<string, number> = {};
      for (const [camId, secs] of Object.entries(transitEdits)) {
        transitUpdates[camId] = secs;
      }

      const hasStartTimeChange = startTime !== (camera.start_time || "");
      const hasTransitChange = Object.keys(transitUpdates).length > 0;

      if (hasStartTimeChange || hasTransitChange) {
        await camerasApi.patch(camera.id, {
          start_time: hasStartTimeChange ? startTime : undefined,
          transit_updates: hasTransitChange ? transitUpdates : undefined,
        });
      }

      setStatusMsg({ text: "Camera configuration saved successfully.", type: "success" });
      onUpdate();
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to save camera changes.";
      setStatusMsg({ text: msg, type: "error" });
    } finally {
      setSaving(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className="space-y-4"
    >
      <Card>
        <CardHeader className="pb-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <CardTitle className="flex items-center gap-2 text-lg">
              <Camera className="h-5 w-5 text-primary" />
              <span>{camera.id}</span>
              <span className="text-ink-faint font-normal">—</span>
              <span className="font-normal text-ink-muted">{camera.location}</span>
            </CardTitle>
            <div className="flex items-center gap-2">
              <Badge variant={camera.is_active ? "success" : "secondary"}>
                {camera.is_active ? "Active" : "Inactive"}
              </Badge>
              {latitude != null && longitude != null && (
                <Badge variant="outline" className="font-mono text-[11px] text-emerald-600 dark:text-emerald-400">
                  <MapPin className="mr-1 h-3 w-3" />
                  GPS Configured
                </Badge>
              )}
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Status alert message */}
          {statusMsg && (
            <div
              className={cn(
                "flex items-center gap-2 rounded-md p-3 text-xs font-medium",
                statusMsg.type === "success"
                  ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border border-emerald-500/20"
                  : "bg-destructive/10 text-destructive border border-destructive/20",
              )}
            >
              {statusMsg.type === "success" ? (
                <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-600 dark:text-emerald-400" />
              ) : (
                <AlertCircle className="h-4 w-4 shrink-0" />
              )}
              <span>{statusMsg.text}</span>
            </div>
          )}

          {/* Location details row */}
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-1.5">
              <Label htmlFor="location-name">Location Name</Label>
              <Input
                id="location-name"
                value={location}
                onChange={(e) => setLocation(e.target.value)}
                placeholder="e.g. Main Entrance Gate"
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="start-time">Recording Start Time (HH:MM:SS)</Label>
              <Input
                id="start-time"
                type="time"
                step="1"
                value={startTime}
                onChange={(e) => setStartTime(e.target.value)}
              />
            </div>
          </div>

          {/* Interactive Google Map Section */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label className="text-sm font-semibold flex items-center gap-1.5">
                  <MapPin className="h-4 w-4 text-primary" />
                  Google Maps Geolocation
                </Label>
                <p className="text-xs text-ink-muted">
                  Drag the marker or click on the map to position this camera.
                </p>
              </div>
              <Button
                variant="ghost"
                size="sm"
                className="h-7 text-xs text-ink-muted hover:text-ink gap-1"
                onClick={handleResetLocation}
                title="Reset to camera's saved position"
              >
                <RotateCcw className="h-3.5 w-3.5" />
                Reset Pin
              </Button>
            </div>

            <CameraMapView
              camera={camera}
              allCameras={allCameras}
              mapsConfig={mapsConfig}
              mapsConfigLoading={mapsConfigLoading}
              latitude={latitude}
              longitude={longitude}
              onChangeLocation={handleCoordinatesChange}
              onSelectCamera={onSelectCamera}
            />

            {/* Coordinates Manual Inputs & Place ID */}
            <div className="grid gap-3 sm:grid-cols-3 pt-1">
              <div className="space-y-1">
                <Label htmlFor="cam-latitude" className="text-xs text-ink-secondary">
                  Latitude (-90 to 90)
                </Label>
                <Input
                  id="cam-latitude"
                  type="number"
                  step="0.000001"
                  min="-90"
                  max="90"
                  value={latitude !== null ? latitude : ""}
                  onChange={(e) => {
                    const val = e.target.value.trim();
                    setLatitude(val === "" ? null : parseFloat(val));
                  }}
                  placeholder="e.g. 30.736768"
                  className="font-mono text-xs"
                />
              </div>

              <div className="space-y-1">
                <Label htmlFor="cam-longitude" className="text-xs text-ink-secondary">
                  Longitude (-180 to 180)
                </Label>
                <Input
                  id="cam-longitude"
                  type="number"
                  step="0.000001"
                  min="-180"
                  max="180"
                  value={longitude !== null ? longitude : ""}
                  onChange={(e) => {
                    const val = e.target.value.trim();
                    setLongitude(val === "" ? null : parseFloat(val));
                  }}
                  placeholder="e.g. 76.776559"
                  className="font-mono text-xs"
                />
              </div>

              <div className="space-y-1">
                <Label htmlFor="cam-place-id" className="text-xs text-ink-secondary">
                  Google Place ID (Optional)
                </Label>
                <Input
                  id="cam-place-id"
                  value={googlePlaceId || ""}
                  onChange={(e) => setGooglePlaceId(e.target.value || null)}
                  placeholder="e.g. ChIJ..."
                  className="font-mono text-xs"
                />
              </div>
            </div>
          </div>

          <Separator />

          {/* Transit times configuration */}
          {camera.connects_to.length > 0 && (
            <div className="space-y-3">
              <div className="space-y-0.5">
                <Label className="text-sm font-semibold">Camera Graph Adjacency</Label>
                <p className="text-xs text-ink-muted">
                  Average walking transit time in seconds to directly adjacent cameras.
                </p>
              </div>
              <div className="grid gap-2 sm:grid-cols-2">
                {camera.connects_to.map((edge) => (
                  <div
                    key={edge.to_camera_id}
                    className="flex items-center gap-3 rounded-md border border-border bg-surface-muted/30 p-2.5"
                  >
                    <ArrowRight className="h-4 w-4 text-ink-muted" />
                    <div className="flex-1 min-w-0">
                      <span className="text-sm font-medium text-ink">{edge.to_camera_id}</span>
                      {edge.notes && (
                        <p className="truncate text-[11px] text-ink-faint">{edge.notes}</p>
                      )}
                    </div>
                    <div className="flex items-center gap-1.5">
                      <Input
                        type="number"
                        min="1"
                        max="3600"
                        className="w-20 font-mono text-xs text-right"
                        defaultValue={edge.avg_transit_sec}
                        onChange={(e) =>
                          setTransitEdits((prev) => ({
                            ...prev,
                            [edge.to_camera_id]: parseInt(e.target.value, 10) || 30,
                          }))
                        }
                      />
                      <span className="text-xs text-ink-muted">s</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Description */}
          <div className="rounded-md border border-border bg-surface-muted/20 p-3 text-xs space-y-1">
            <span className="font-semibold text-ink-secondary">Description & Notes</span>
            <p className="text-ink-muted leading-relaxed">{camera.description || "No description provided."}</p>
          </div>

          {/* Actions */}
          <div className="flex items-center justify-between pt-2">
            <span className="text-xs text-ink-muted">
              PUT /api/cameras/{camera.id}/location &amp; PATCH /api/cameras/{camera.id}
            </span>
            <Button onClick={handleSave} disabled={saving} className="gap-2">
              <Save className="h-4 w-4" />
              {saving ? "Saving Changes…" : "Save Changes"}
            </Button>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}

// ── Google Map View Component ───────────────────────────────────────────────

function CameraMapView({
  camera,
  allCameras,
  mapsConfig,
  mapsConfigLoading,
  latitude,
  longitude,
  onChangeLocation,
  onSelectCamera,
}: {
  camera: CameraWithGraph;
  allCameras: CameraWithGraph[];
  mapsConfig: CameraMapsConfig | null;
  mapsConfigLoading: boolean;
  latitude: number | null;
  longitude: number | null;
  onChangeLocation: (lat: number, lng: number, placeId?: string | null, placeName?: string) => void;
  onSelectCamera: (id: string) => void;
}) {
  const mapContainerRef = React.useRef<HTMLDivElement>(null);
  const searchInputRef = React.useRef<HTMLInputElement>(null);

  const mapInstanceRef = React.useRef<google.maps.Map | null>(null);
  const activeMarkerRef = React.useRef<google.maps.Marker | null>(null);
  const otherMarkersRef = React.useRef<google.maps.Marker[]>([]);

  const [mapLoaded, setMapLoaded] = React.useState(false);
  const [mapError, setMapError] = React.useState<string | null>(null);
  const [locating, setLocating] = React.useState(false);

  const isConfigured = Boolean(mapsConfig?.configured && mapsConfig?.google_maps_api_key);

  // Initialize Google Maps script and map instance
  React.useEffect(() => {
    if (mapsConfigLoading || !isConfigured || !mapsConfig?.google_maps_api_key) return;

    let isSubscribed = true;

    async function initMap() {
      try {
        setOptions({
          key: mapsConfig!.google_maps_api_key,
          v: "weekly",
        });

        // Load Maps & Marker libraries
        const [{ Map }, { Marker }] = await Promise.all([
          importLibrary("maps") as Promise<google.maps.MapsLibrary>,
          importLibrary("marker") as Promise<google.maps.MarkerLibrary>,
        ]);

        if (!isSubscribed || !mapContainerRef.current) return;

        const centerLat = latitude ?? DEFAULT_CENTER.lat;
        const centerLng = longitude ?? DEFAULT_CENTER.lng;
        const initialZoom = latitude != null && longitude != null ? 16 : 14;

        const map = new Map(mapContainerRef.current, {
          center: { lat: centerLat, lng: centerLng },
          zoom: initialZoom,
          mapTypeControl: true,
          streetViewControl: false,
          fullscreenControl: true,
          zoomControl: true,
        });
        mapInstanceRef.current = map;

        // Selected camera marker (draggable)
        const marker = new Marker({
          position: { lat: centerLat, lng: centerLng },
          map,
          draggable: true,
          title: `${camera.id} (${camera.location})`,
          animation: google.maps.Animation.DROP,
        });
        activeMarkerRef.current = marker;

        // Marker drag handler
        marker.addListener("dragend", () => {
          const pos = marker.getPosition();
          if (pos) {
            onChangeLocation(pos.lat(), pos.lng());
          }
        });

        // Map click handler (drop / move pin)
        map.addListener("click", (e: google.maps.MapMouseEvent) => {
          if (e.latLng) {
            const lat = e.latLng.lat();
            const lng = e.latLng.lng();
            marker.setPosition({ lat, lng });
            onChangeLocation(lat, lng);
          }
        });

        // Other cameras markers
        otherMarkersRef.current.forEach((m) => m.setMap(null));
        otherMarkersRef.current = [];

        allCameras
          .filter((c) => c.id !== camera.id && c.latitude != null && c.longitude != null)
          .forEach((otherCam) => {
            const otherMarker = new Marker({
              position: { lat: otherCam.latitude!, lng: otherCam.longitude! },
              map,
              title: `${otherCam.id} — ${otherCam.location}`,
              opacity: 0.7,
              label: {
                text: otherCam.id,
                color: "#1e293b",
                fontSize: "11px",
                fontWeight: "bold",
              },
            });
            otherMarker.addListener("click", () => {
              onSelectCamera(otherCam.id);
            });
            otherMarkersRef.current.push(otherMarker);
          });

        // Setup Places Autocomplete if available
        try {
          const placesLib = (await importLibrary("places")) as google.maps.PlacesLibrary;
          if (isSubscribed && searchInputRef.current) {
            const autocomplete = new placesLib.Autocomplete(searchInputRef.current, {
              fields: ["geometry", "formatted_address", "name", "place_id"],
            });
            autocomplete.bindTo("bounds", map);
            autocomplete.addListener("place_changed", () => {
              const place = autocomplete.getPlace();
              if (place.geometry?.location) {
                const lat = place.geometry.location.lat();
                const lng = place.geometry.location.lng();
                map.panTo({ lat, lng });
                map.setZoom(17);
                marker.setPosition({ lat, lng });
                onChangeLocation(lat, lng, place.place_id, place.name || place.formatted_address);
              }
            });
          }
        } catch {
          // Places autocomplete unavailable or not enabled on key — map click & drag still work
        }

        if (isSubscribed) {
          setMapLoaded(true);
        }
      } catch (err) {
        if (isSubscribed) {
          const msg = err instanceof Error ? err.message : "Failed to load Google Maps";
          setMapError(msg);
        }
      }
    }

    initMap();

    return () => {
      isSubscribed = false;
      if (activeMarkerRef.current) {
        activeMarkerRef.current.setMap(null);
      }
      otherMarkersRef.current.forEach((m) => m.setMap(null));
      otherMarkersRef.current = [];
    };
  }, [camera.id, isConfigured, mapsConfig?.google_maps_api_key]);

  // Sync marker position when latitude or longitude props change (e.g. from manual inputs)
  React.useEffect(() => {
    if (activeMarkerRef.current && mapInstanceRef.current && latitude !== null && longitude !== null) {
      const curPos = activeMarkerRef.current.getPosition();
      if (
        !curPos ||
        Math.abs(curPos.lat() - latitude) > 0.000001 ||
        Math.abs(curPos.lng() - longitude) > 0.000001
      ) {
        const nextPos = { lat: latitude, lng: longitude };
        activeMarkerRef.current.setPosition(nextPos);
        mapInstanceRef.current.panTo(nextPos);
      }
    }
  }, [latitude, longitude]);

  // Geolocation helper: Use Current Location
  const handleLocateMe = () => {
    if (!navigator.geolocation) {
      alert("Geolocation is not supported by your browser.");
      return;
    }
    setLocating(true);
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setLocating(false);
        const lat = pos.coords.latitude;
        const lng = pos.coords.longitude;
        if (mapInstanceRef.current) {
          mapInstanceRef.current.panTo({ lat, lng });
          mapInstanceRef.current.setZoom(17);
        }
        if (activeMarkerRef.current) {
          activeMarkerRef.current.setPosition({ lat, lng });
        }
        onChangeLocation(lat, lng);
      },
      (err) => {
        setLocating(false);
        alert(`Could not retrieve location: ${err.message}`);
      },
      { timeout: 10000 },
    );
  };

  // 1. Loading maps configuration
  if (mapsConfigLoading) {
    return <Skeleton className="h-[360px] w-full rounded-lg" />;
  }

  // 2. Maps not configured in backend/.env
  if (!isConfigured) {
    return (
      <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-border bg-surface-muted/20 p-8 text-center">
        <AlertCircle className="mb-2 h-7 w-7 text-amber-500" />
        <h4 className="text-sm font-semibold text-ink">Google Maps API Key Not Configured</h4>
        <p className="mt-1 max-w-md text-xs text-ink-muted">
          Add <code className="rounded bg-surface-muted px-1.5 py-0.5 font-mono text-[11px]">GOOGLE_MAPS_API_KEY</code> to your <code className="rounded bg-surface-muted px-1.5 py-0.5 font-mono text-[11px]">.env</code> file to enable interactive maps and place search.
        </p>
      </div>
    );
  }

  // 3. Error loading script
  if (mapError) {
    return (
      <div className="flex flex-col items-center justify-center rounded-lg border border-destructive/20 bg-destructive/5 p-6 text-center">
        <AlertCircle className="mb-2 h-6 w-6 text-destructive" />
        <h4 className="text-sm font-semibold text-destructive">Google Maps Error</h4>
        <p className="mt-1 max-w-md text-xs text-ink-muted">{mapError}</p>
        <p className="mt-2 text-[11px] text-ink-faint">
          You can still enter latitude and longitude coordinates manually below.
        </p>
      </div>
    );
  }

  // 4. Render interactive Map
  return (
    <div className="space-y-2">
      {/* Search box overlay / input */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-ink-faint" />
        <Input
          ref={searchInputRef}
          type="text"
          placeholder="Search places or addresses with Google Places…"
          className="pl-8 text-xs h-9 bg-surface"
        />
      </div>

      {/* Map container */}
      <div className="relative h-[360px] w-full overflow-hidden rounded-lg border border-border bg-surface-muted">
        {!mapLoaded && (
          <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-2 bg-surface/80 backdrop-blur-sm">
            <Skeleton className="h-full w-full absolute inset-0" />
            <span className="relative z-20 text-xs font-medium text-ink-muted">
              Loading Google Maps…
            </span>
          </div>
        )}
        <div ref={mapContainerRef} className="h-full w-full" />

        {/* Floating Toolbar inside Map */}
        <div className="absolute bottom-3 left-3 z-10 flex items-center gap-2">
          <Button
            type="button"
            variant="secondary"
            size="sm"
            onClick={handleLocateMe}
            disabled={locating}
            className="h-7 gap-1 bg-surface/90 text-[11px] shadow-sm backdrop-blur hover:bg-surface border border-border"
          >
            <Crosshair className="h-3 w-3" />
            {locating ? "Locating…" : "My Location"}
          </Button>

          {latitude != null && longitude != null && (
            <a
              href={`https://www.google.com/maps/search/?api=1&query=${latitude},${longitude}`}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex h-7 items-center gap-1 rounded-md border border-border bg-surface/90 px-2 text-[11px] font-medium text-ink-secondary shadow-sm backdrop-blur hover:bg-surface hover:text-ink"
              title="Open in Google Maps"
            >
              <ExternalLink className="h-3 w-3" />
              Open in Maps
            </a>
          )}
        </div>
      </div>
    </div>
  );
}
