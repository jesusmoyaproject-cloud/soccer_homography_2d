"""
Módulo de Post-Procesamiento Tabular y Cosido Global de Tracklets (Pasada 2):
Implementa `GlobalTrackletStitcher` sobre los registros tabulares generados en la Pasada 1:
  1. Purificación y división de fragmentos en cortes de cámara o saltos cinemáticos imposibles.
  2. Propagación retroactiva y hacia adelante de dorsales leídos (votación ponderada por confianza)
     y equipo a todo el tracklet.
  3. Cosido Global de Tracklets (Global Tracklet Stitching) mediante Asignación Húngara usando:
     Distancia 2D en cancha + Tiempo (Δt) + Equipo + Color CIE-LAB + Dorsal.
  4. Interpolación 2D de oclusiones breves y exportación tabular a CSV.
"""

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.spatial_geometry import apollonius_reach_cost


@dataclass
class TrackletSummary:
    """Resumen estadístico, espacial y de identidad de un fragmento de trayectoria (tracklet)."""
    fragment_id: int
    raw_track_id: int
    start_frame: int
    end_frame: int
    num_frames: int
    team_id: int
    team_label: str
    role: str
    color_bgr: Tuple[int, int, int]
    mean_lab: Optional[np.ndarray]
    start_pos_m: Optional[Tuple[float, float]]
    end_pos_m: Optional[Tuple[float, float]]
    end_vel_mps: Tuple[float, float] = (0.0, 0.0)
    confirmed_jersey: Optional[str] = None
    jersey_conf_sum: float = 0.0
    jersey_votes: Dict[str, float] = field(default_factory=dict)
    row_indices: List[int] = field(default_factory=list)


class GlobalTrackletStitcher:
    """
    Algoritmo de Pasada 2 para Cosido Global de Tracklets (Global Tracklet Stitching)
    y propagación completa de dorsales sobre datos tabulares.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, fps: float = 25.0):
        cfg = (config or {}).get("tracking_2d", {}).get("stitching", {})
        self.fps = max(1.0, float(fps))
        self.max_speed_mps = float(cfg.get("max_speed_mps", 12.0))
        self.max_stitch_gap_sec = float(cfg.get("max_stitch_gap_sec", 10.0))
        self.max_stitch_dist_m = float(cfg.get("max_stitch_dist_m", 18.0))
        self.min_jersey_conf_sum = float(cfg.get("min_jersey_conf_sum", 0.50))
        self.interpolate_max_gap_frames = int(cfg.get("interpolate_max_gap_frames", 12))

    def _split_into_atomic_fragments(
        self, records: List[Dict[str, Any]]
    ) -> Dict[int, List[int]]:
        """
        Agrupa las filas por `online_track_id` y divide un tracklet en sub-fragmentos si:
        - Ocurrió un corte de cámara (`shot_cut == True`) en medio del tracklet.
        - Hay un salto espacial en cancha > max_speed_mps * dt + 3.5 m entre dos cuadros consecutivos.
        """
        by_raw: Dict[int, List[int]] = defaultdict(list)
        for row_idx, rec in enumerate(records):
            tid = int(rec.get("online_track_id", rec.get("raw_track_id", 0)))
            by_raw[tid].append(row_idx)

        fragments: Dict[int, List[int]] = {}
        next_frag_id = 1

        for tid, idxs in by_raw.items():
            idxs.sort(key=lambda idx: int(records[idx]["frame"]))
            current_frag: List[int] = []

            for k, r_idx in enumerate(idxs):
                rec = records[r_idx]
                if not current_frag:
                    current_frag.append(r_idx)
                    continue

                prev_rec = records[current_frag[-1]]
                dt = max(1.0 / self.fps, (int(rec["frame"]) - int(prev_rec["frame"])) / self.fps)

                must_split = False
                if rec.get("shot_cut", False) and int(rec["frame"]) > int(prev_rec["frame"]):
                    must_split = True
                elif (
                    rec.get("pitch_x_m") is not None
                    and prev_rec.get("pitch_x_m") is not None
                ):
                    dx = float(rec["pitch_x_m"]) - float(prev_rec["pitch_x_m"])
                    dy = float(rec["pitch_y_m"]) - float(prev_rec["pitch_y_m"])
                    jump_m = float(np.hypot(dx, dy))
                    if jump_m > (self.max_speed_mps * dt + 4.0):
                        must_split = True

                if must_split:
                    fragments[next_frag_id] = current_frag
                    next_frag_id += 1
                    current_frag = [r_idx]
                else:
                    current_frag.append(r_idx)

            if current_frag:
                fragments[next_frag_id] = current_frag
                next_frag_id += 1

        return fragments

    def _build_fragment_summaries(
        self,
        records: List[Dict[str, Any]],
        fragments: Dict[int, List[int]],
    ) -> Dict[int, TrackletSummary]:
        """Calcula los estadísticos de posición 2D, velocidad, equipo, color y dorsal de cada fragmento."""
        summaries: Dict[int, TrackletSummary] = {}

        for frag_id, row_idxs in fragments.items():
            first_rec = records[row_idxs[0]]
            last_rec = records[row_idxs[-1]]
            raw_tid = int(first_rec.get("raw_track_id", first_rec.get("online_track_id", frag_id)))

            # Votación de equipo y rol
            team_counts: Dict[int, int] = defaultdict(int)
            role_counts: Dict[str, int] = defaultdict(int)
            bgr_list: List[Tuple[int, int, int]] = []
            lab_list: List[List[float]] = []
            jersey_votes: Dict[str, float] = defaultdict(float)
            valid_positions: List[Tuple[int, float, float]] = []

            for r_idx in row_idxs:
                r = records[r_idx]
                team_counts[int(r.get("team_id", 1))] += 1
                role_counts[str(r.get("role", "Player"))] += 1
                bgr_list.append((int(r.get("color_b", 255)), int(r.get("color_g", 190)), int(r.get("color_r", 0))))

                if r.get("lab_l") is not None and r.get("lab_a") is not None and r.get("lab_b") is not None:
                    lab_list.append([float(r["lab_l"]), float(r["lab_a"]), float(r["lab_b"])])

                j_raw = r.get("jersey_ocr_raw")
                j_conf = float(r.get("jersey_ocr_conf", 0.0))
                if j_raw is not None and str(j_raw).strip() != "" and j_conf > 0.0:
                    jersey_votes[str(j_raw).strip()] += j_conf

                if r.get("pitch_x_m") is not None and r.get("pitch_y_m") is not None:
                    valid_positions.append((int(r["frame"]), float(r["pitch_x_m"]), float(r["pitch_y_m"])))

            best_team = max(team_counts.items(), key=lambda kv: kv[1])[0]
            best_role = max(role_counts.items(), key=lambda kv: kv[1])[0]
            team_label_map = {1: "Team_A", 2: "Team_B", 3: "GK", 4: "Referee"}
            team_label = team_label_map.get(best_team, "Team_A")

            med_bgr = np.median(np.array(bgr_list), axis=0)
            color_bgr = (int(med_bgr[0]), int(med_bgr[1]), int(med_bgr[2]))
            mean_lab = np.mean(np.array(lab_list, dtype=np.float32), axis=0) if lab_list else None

            confirmed_jersey = None
            jersey_conf_sum = 0.0
            if jersey_votes:
                best_j, best_score = max(jersey_votes.items(), key=lambda kv: kv[1])
                if best_score >= self.min_jersey_conf_sum:
                    confirmed_jersey = best_j
                    jersey_conf_sum = float(best_score)

            start_pos_m = (valid_positions[0][1], valid_positions[0][2]) if valid_positions else None
            end_pos_m = (valid_positions[-1][1], valid_positions[-1][2]) if valid_positions else None

            # Estimar velocidad de salida en metros/s usando los últimos puntos válidos del fragmento
            end_vel = (0.0, 0.0)
            if len(valid_positions) >= 2:
                k_back = max(0, len(valid_positions) - min(6, len(valid_positions)))
                f0, x0, y0 = valid_positions[k_back]
                f1, x1, y1 = valid_positions[-1]
                dt_v = (f1 - f0) / self.fps
                if dt_v > 0.02:
                    vx = float(np.clip((x1 - x0) / dt_v, -self.max_speed_mps, self.max_speed_mps))
                    vy = float(np.clip((y1 - y0) / dt_v, -self.max_speed_mps, self.max_speed_mps))
                    end_vel = (vx, vy)

            summaries[frag_id] = TrackletSummary(
                fragment_id=frag_id,
                raw_track_id=raw_tid,
                start_frame=int(first_rec["frame"]),
                end_frame=int(last_rec["frame"]),
                num_frames=len(row_idxs),
                team_id=best_team,
                team_label=team_label,
                role=best_role,
                color_bgr=color_bgr,
                mean_lab=mean_lab,
                start_pos_m=start_pos_m,
                end_pos_m=end_pos_m,
                end_vel_mps=end_vel,
                confirmed_jersey=confirmed_jersey,
                jersey_conf_sum=jersey_conf_sum,
                jersey_votes=dict(jersey_votes),
                row_indices=row_idxs,
            )

        return summaries

    def _compute_stitch_cost(
        self,
        frag_a: TrackletSummary,
        frag_b: TrackletSummary,
    ) -> float:
        """
        Calcula el costo de coser el final del fragmento A con el inicio del fragmento B.
        Retorna >= 1e6 si la unión es física o semánticamente inválida.
        """
        INF = 1e6
        # 1. No pueden solaparse en el tiempo; B debe iniciar después de que termina A
        frame_gap = frag_b.start_frame - frag_a.end_frame
        if frame_gap <= 0:
            return INF

        dt_sec = frame_gap / self.fps

        # 2. Gating estricto por Equipo
        if frag_a.team_id != frag_b.team_id:
            return INF

        # 3. Gating por Dorsal: si ambos tienen dorsal confirmado y son distintos -> imposible
        same_confirmed_jersey = False
        if frag_a.confirmed_jersey and frag_b.confirmed_jersey:
            if frag_a.confirmed_jersey != frag_b.confirmed_jersey:
                return INF
            same_confirmed_jersey = True

        # Si no comparten dorsal confirmado, limitar el hueco temporal a max_stitch_gap_sec
        if not same_confirmed_jersey and dt_sec > self.max_stitch_gap_sec:
            return INF

        # 4. Distancia en el plano 2D de la cancha con Región Cinemática de Apolonio
        dist_cost = 0.0
        if frag_a.end_pos_m is not None and frag_b.start_pos_m is not None:
            apol_dist_m = apollonius_reach_cost(
                frag_a.end_pos_m,
                frag_a.end_vel_mps,
                frag_b.start_pos_m,
                dt_sec=dt_sec,
                max_speed_mps=self.max_speed_mps,
            )
            raw_dist_m = float(
                np.hypot(frag_b.start_pos_m[0] - frag_a.end_pos_m[0], frag_b.start_pos_m[1] - frag_a.end_pos_m[1])
            )
            eff_dist_m = min(apol_dist_m, raw_dist_m)

            max_reachable_m = self.max_speed_mps * dt_sec + 3.5
            if not same_confirmed_jersey:
                max_reachable_m = min(max_reachable_m, self.max_stitch_dist_m)

            if eff_dist_m > max_reachable_m:
                return INF

            dist_cost = eff_dist_m / max(2.0, self.max_stitch_dist_m)
        elif not same_confirmed_jersey:
            # Si no hay coordenadas 2D ni dorsal confirmado, no arriesgar uniones a ciegas
            return INF

        # 5. Similitud cromática en espacio CIE-LAB
        color_cost = 0.0
        if frag_a.mean_lab is not None and frag_b.mean_lab is not None:
            lab_diff = float(np.linalg.norm(frag_a.mean_lab - frag_b.mean_lab))
            if lab_diff > 45.0 and not same_confirmed_jersey:
                return INF
            color_cost = lab_diff / 45.0

        # 6. Penalización suave por hueco temporal + bonificación fuerte por coincidencia de dorsal
        time_cost = 0.35 * (dt_sec / max(1.0, self.max_stitch_gap_sec))
        jersey_bonus = -2.0 if same_confirmed_jersey else 0.0

        return float(dist_cost + color_cost + time_cost + jersey_bonus)

    def stitch_and_propagate(
        self,
        records: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Ejecuta la Pasada 2 completa sobre la tabla de la Pasada 1:
        1. Divide fragmentos rotos por cortes o saltos.
        2. Consolida dorsales y equipos por fragmento.
        3. Une globalmente los fragmentos (Global Tracklet Stitching) con Asignación Húngara.
        4. Propaga el `global_track_id` y el `stitched_jersey` a todos los cuadros de la cadena.
        5. Interpola huecos cortos en el plano 2D.
        """
        if not records:
            return [], {"raw_tracklets": 0, "stitched_players": 0, "merges_performed": 0, "jerseys_assigned": 0}

        # Paso 1 y 2: Fragmentos atómicos y resúmenes
        fragments = self._split_into_atomic_fragments(records)
        summaries = self._build_fragment_summaries(records, fragments)
        frag_ids = sorted(summaries.keys(), key=lambda fid: (summaries[fid].start_frame, fid))

        # Paso 3: Grafo de Cosido Global mediante Asignación Húngara iterativa
        n = len(frag_ids)
        next_link: Dict[int, int] = {}
        prev_link: Dict[int, int] = {}

        if n >= 2:
            cost_matrix = np.full((n, n), 1e6, dtype=np.float64)
            for i, fid_a in enumerate(frag_ids):
                for j, fid_b in enumerate(frag_ids):
                    if i == j:
                        continue
                    c = self._compute_stitch_cost(summaries[fid_a], summaries[fid_b])
                    if c < 1e5:
                        cost_matrix[i, j] = c

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 1e5:
                    fid_a = frag_ids[r]
                    fid_b = frag_ids[c]
                    next_link[fid_a] = fid_b
                    prev_link[fid_b] = fid_a

        # Paso 4: Construir cadenas conexas de fragmentos (caminos desde raíces sin prev_link)
        chains: List[List[int]] = []
        visited = set()
        for fid in frag_ids:
            if fid not in prev_link and fid not in visited:
                chain = []
                curr = fid
                while curr is not None and curr not in visited:
                    visited.add(curr)
                    chain.append(curr)
                    curr = next_link.get(curr)
                chains.append(chain)

        # Añadir cualquier ciclo remanente por seguridad
        for fid in frag_ids:
            if fid not in visited:
                chains.append([fid])
                visited.add(fid)

        # Ordenar las identidades globales por primer frame de aparición y duración
        chains.sort(key=lambda ch: (summaries[ch[0]].start_frame, -sum(summaries[f].num_frames for f in ch)))

        stitched_records = [dict(r) for r in records]
        jerseys_assigned_count = 0
        merges_count = sum(len(ch) - 1 for ch in chains)
        identities_manifest: List[Dict[str, Any]] = []

        for global_id, chain in enumerate(chains, start=1):
            # Combinar todos los votos de dorsal y equipo a lo largo de todos los fragmentos cosidos
            combined_jersey_votes: Dict[str, float] = defaultdict(float)
            team_votes: Dict[int, int] = defaultdict(int)
            bgr_samples: List[Tuple[int, int, int]] = []
            raw_tids_ordered: List[int] = []

            for fid in chain:
                s = summaries[fid]
                team_votes[s.team_id] += s.num_frames
                bgr_samples.append(s.color_bgr)
                if s.raw_track_id not in raw_tids_ordered:
                    raw_tids_ordered.append(int(s.raw_track_id))
                for j_num, j_score in s.jersey_votes.items():
                    combined_jersey_votes[j_num] += j_score

            final_team_id = max(team_votes.items(), key=lambda kv: kv[1])[0]
            team_label_map = {1: "Team_A", 2: "Team_B", 3: "GK", 4: "Referee"}
            final_team_label = team_label_map.get(final_team_id, "Team_A")
            med_bgr = np.median(np.array(bgr_samples), axis=0)
            final_bgr = (int(med_bgr[0]), int(med_bgr[1]), int(med_bgr[2]))

            final_jersey: Optional[str] = None
            if combined_jersey_votes:
                best_j, best_score = max(combined_jersey_votes.items(), key=lambda kv: kv[1])
                if best_score >= self.min_jersey_conf_sum:
                    final_jersey = best_j
                    jerseys_assigned_count += 1

            # Propagar el global_id, equipo y dorsal confirmado a TODAS las filas de la cadena (Pasada 2)
            chain_row_indices: List[int] = []
            for fid in chain:
                chain_row_indices.extend(summaries[fid].row_indices)

            chain_row_indices.sort(key=lambda idx: int(stitched_records[idx]["frame"]))
            speeds_in_chain: List[float] = []
            confs_in_chain: List[float] = []
            for r_idx in chain_row_indices:
                rec = stitched_records[r_idx]
                rec["global_track_id"] = int(global_id)
                rec["team_id"] = int(final_team_id)
                rec["team_label"] = final_team_label
                rec["color_b"] = final_bgr[0]
                rec["color_g"] = final_bgr[1]
                rec["color_r"] = final_bgr[2]
                rec["stitched_jersey"] = final_jersey
                rec["is_interpolated"] = False
                speeds_in_chain.append(float(rec.get("speed_mps", 0.0)))
                confs_in_chain.append(float(rec.get("conf", 0.75)))

            # Construir metadatos de enlaces para identity_manifest.json (excluyendo auto-enlaces del mismo raw_track_id)
            links_info: List[Dict[str, Any]] = []
            for k_idx in range(len(chain) - 1):
                sa = summaries[chain[k_idx]]
                sb = summaries[chain[k_idx + 1]]
                if int(sa.raw_track_id) == int(sb.raw_track_id):
                    continue
                gap_f = max(0, int(sb.start_frame - sa.end_frame))
                gap_s = round(gap_f / self.fps, 4)
                sp_dist = (
                    round(float(np.hypot(sb.start_pos_m[0] - sa.end_pos_m[0], sb.start_pos_m[1] - sa.end_pos_m[1])), 2)
                    if (sa.end_pos_m is not None and sb.start_pos_m is not None)
                    else 0.0
                )
                raw_cost = self._compute_stitch_cost(sa, sb)
                link_score = round(float(np.clip(1.0 / (1.0 + max(0.0, raw_cost)), 0.50, 0.99)), 4)
                links_info.append(
                    {
                        "from_track_id": int(sa.raw_track_id),
                        "to_track_id": int(sb.raw_track_id),
                        "score": link_score,
                        "gap_frames": gap_f,
                        "gap_seconds": gap_s,
                        "spatial_distance_m": sp_dist,
                    }
                )

            first_f = int(summaries[chain[0]].start_frame)
            last_f = int(summaries[chain[-1]].end_frame)
            avg_spd = round(float(np.mean(speeds_in_chain)), 2) if speeds_in_chain else 0.0
            avg_conf = round(float(np.mean(confs_in_chain)), 4) if confs_in_chain else 0.80
            is_multi = len(raw_tids_ordered) > 1
            identities_manifest.append(
                {
                    "identity_id": f"ID_{global_id:04d}",
                    "global_track_id": int(global_id),
                    "status": "CONFIRMED" if (not is_multi or avg_conf >= 0.65) else "NEEDS_REVIEW",
                    "creation_method": "AUTOMATIC",
                    "confidence_score": avg_conf,
                    "track_ids": [int(global_id)] if not is_multi else ([int(global_id)] + [t for t in raw_tids_ordered if t != int(global_id)]),
                    "raw_track_ids": raw_tids_ordered,
                    "segment_count": len(raw_tids_ordered),
                    "frame_span": {
                        "first_frame": first_f,
                        "last_frame": last_f,
                        "total_active_frames": len(chain_row_indices),
                    },
                    "kinematics": {
                        "avg_velocity_ms": avg_spd,
                    },
                    "links": links_info,
                    "enrichment": {
                        "team_id": int(final_team_id),
                        "shirt_number": final_jersey,
                    },
                }
            )

            # Paso 5: Interpolación lineal en el plano 2D para huecos breves entre fragmentos de la misma cadena
            interpolated_rows: List[Dict[str, Any]] = []
            for k in range(len(chain_row_indices) - 1):
                r_curr = stitched_records[chain_row_indices[k]]
                r_next = stitched_records[chain_row_indices[k + 1]]
                f_curr = int(r_curr["frame"])
                f_next = int(r_next["frame"])
                gap = f_next - f_curr

                if (
                    1 < gap <= self.interpolate_max_gap_frames
                    and not r_next.get("shot_cut", False)
                    and r_curr.get("pitch_x_m") is not None
                    and r_next.get("pitch_x_m") is not None
                ):
                    for step in range(1, gap):
                        alpha = float(step) / float(gap)
                        interp_row = dict(r_curr)
                        interp_row["frame"] = f_curr + step
                        interp_row["bbox_x1"] = round((1 - alpha) * r_curr["bbox_x1"] + alpha * r_next["bbox_x1"], 2)
                        interp_row["bbox_y1"] = round((1 - alpha) * r_curr["bbox_y1"] + alpha * r_next["bbox_y1"], 2)
                        interp_row["bbox_x2"] = round((1 - alpha) * r_curr["bbox_x2"] + alpha * r_next["bbox_x2"], 2)
                        interp_row["bbox_y2"] = round((1 - alpha) * r_curr["bbox_y2"] + alpha * r_next["bbox_y2"], 2)
                        interp_row["foot_px_x"] = round(
                            (1 - alpha) * r_curr["foot_px_x"] + alpha * r_next["foot_px_x"], 2
                        )
                        interp_row["foot_px_y"] = round(
                            (1 - alpha) * r_curr["foot_px_y"] + alpha * r_next["foot_px_y"], 2
                        )
                        interp_row["pitch_x_m"] = round(
                            (1 - alpha) * float(r_curr["pitch_x_m"]) + alpha * float(r_next["pitch_x_m"]), 3
                        )
                        interp_row["pitch_y_m"] = round(
                            (1 - alpha) * float(r_curr["pitch_y_m"]) + alpha * float(r_next["pitch_y_m"]), 3
                        )
                        interp_row["is_interpolated"] = True
                        interpolated_rows.append(interp_row)

            stitched_records.extend(interpolated_rows)

        stitched_records.sort(key=lambda r: (int(r["frame"]), int(r.get("global_track_id", 0))))

        stats = {
            "raw_tracklets": len(fragments),
            "stitched_players": len(chains),
            "merges_performed": merges_count,
            "jerseys_assigned": jerseys_assigned_count,
            "total_rows": len(stitched_records),
            "identities_manifest": identities_manifest,
        }
        return stitched_records, stats

    @staticmethod
    def build_frame_lookup(
        stitched_records: List[Dict[str, Any]]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Convierte la tabla cosida de la Pasada 2 en un diccionario indexado por `frame`
        listo para ser consumido directamente por `Tactical2DRenderer`.
        """
        by_frame: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for rec in stitched_records:
            f_idx = int(rec["frame"])
            pm = (
                (float(rec["pitch_x_m"]), float(rec["pitch_y_m"]))
                if rec.get("pitch_x_m") is not None and rec.get("pitch_y_m") is not None
                else None
            )
            by_frame[f_idx].append(
                {
                    "id": int(rec.get("global_track_id", rec.get("online_track_id", 1))),
                    "raw_track_id": int(rec.get("raw_track_id", 1)),
                    "class_id": int(rec.get("class_id", 2)),
                    "role": str(rec.get("role", "Player")),
                    "conf": float(rec.get("conf", 0.5)),
                    "bbox": [
                        float(rec["bbox_x1"]),
                        float(rec["bbox_y1"]),
                        float(rec["bbox_x2"]),
                        float(rec["bbox_y2"]),
                    ],
                    "foot_px": (float(rec["foot_px_x"]), float(rec["foot_px_y"])),
                    "pitch_m": pm,
                    "speed_mps": float(rec.get("speed_mps", 0.0)),
                    "vx_mps": float(rec.get("vx_mps", 0.0)),
                    "vy_mps": float(rec.get("vy_mps", 0.0)),
                    "team_id": int(rec.get("team_id", 1)),
                    "team_label": str(rec.get("team_label", "Team_A")),
                    "color_bgr": (
                        int(rec.get("color_b", 255)),
                        int(rec.get("color_g", 190)),
                        int(rec.get("color_r", 0)),
                    ),
                    "jersey_number": rec.get("stitched_jersey") or rec.get("jersey_online"),
                    "is_interpolated": bool(rec.get("is_interpolated", False)),
                }
            )
        return by_frame

    @staticmethod
    def export_to_csv(records: List[Dict[str, Any]], csv_path: Union[str, Path]) -> Path:
        """Exporta los registros tabulares (crudos o cosidos) a un archivo CSV."""
        out_path = Path(csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not records:
            out_path.write_text("", encoding="utf-8")
            return out_path

        fieldnames = list(records[0].keys())
        for r in records:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        return out_path

