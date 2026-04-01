# infra/main.tf
# Provisions the full GCP stack for rsd-convert.
#
# Resources:
#   - Artifact Registry repo (Docker images)
#   - Cloud Run service
#   - Cloud SQL (Postgres 15) + private IP via VPC connector
#   - Secret Manager secrets (API keys + DB URL)
#   - Cloud Build trigger (GitHub)
#   - IAM bindings
#
# Usage:
#   cd infra
#   terraform init
#   terraform apply -var="project_id=YOUR_PROJECT" -var="github_owner=YOUR_GH_USER"

terraform {
  required_version = ">= 1.6"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
  }
}

# ── Variables ────────────────────────────────────────────────────────────────
variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "Primary GCP region"
  type        = string
  default     = "australia-southeast1"
}

variable "github_owner" {
  description = "GitHub username / org that owns rsd-convert"
  type        = string
}

variable "github_repo" {
  description = "GitHub repository name"
  type        = string
  default     = "rsd-convert"
}

variable "db_tier" {
  description = "Cloud SQL machine tier"
  type        = string
  default     = "db-g1-small"
}

# ── Providers ────────────────────────────────────────────────────────────────
provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# ── Enable APIs ───────────────────────────────────────────────────────────────
locals {
  apis = [
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "secretmanager.googleapis.com",
    "sqladmin.googleapis.com",
    "servicenetworking.googleapis.com",
    "vpcaccess.googleapis.com",
  ]
}

resource "google_project_service" "apis" {
  for_each = toset(local.apis)
  service  = each.value
  disable_on_destroy = false
}

# ── VPC & Serverless VPC connector ───────────────────────────────────────────
resource "google_compute_network" "vpc" {
  name                    = "rsd-vpc"
  auto_create_subnetworks = false
  depends_on = [google_project_service.apis]
}

resource "google_compute_subnetwork" "subnet" {
  name          = "rsd-subnet"
  ip_cidr_range = "10.8.0.0/24"
  region        = var.region
  network       = google_compute_network.vpc.id
}

resource "google_vpc_access_connector" "connector" {
  provider      = google-beta
  name          = "rsd-vpc-connector"
  region        = var.region
  subnet {
    name = google_compute_subnetwork.subnet.name
  }
  machine_type  = "e2-micro"
  min_instances = 2
  max_instances = 5
  depends_on    = [google_project_service.apis]
}

# ── Cloud SQL (Postgres 15) ───────────────────────────────────────────────────
resource "random_password" "db_password" {
  length  = 32
  special = false
}

resource "google_sql_database_instance" "postgres" {
  name             = "rsd-postgres"
  database_version = "POSTGRES_15"
  region           = var.region
  deletion_protection = true

  settings {
    tier              = var.db_tier
    availability_type = "ZONAL"
    disk_autoresize   = true
    disk_size         = 10

    backup_configuration {
      enabled            = true
      start_time         = "03:00"
      binary_log_enabled = false
      backup_retention_settings {
        retained_backups = 7
      }
    }

    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.vpc.id
    }

    database_flags {
      name  = "max_connections"
      value = "100"
    }
  }

  depends_on = [google_project_service.apis]
}

resource "google_sql_database" "rsd_db" {
  name     = "rsd"
  instance = google_sql_database_instance.postgres.name
}

resource "google_sql_user" "app_user" {
  name     = "rsd_app"
  instance = google_sql_database_instance.postgres.name
  password = random_password.db_password.result
}

# ── Artifact Registry ─────────────────────────────────────────────────────────
resource "google_artifact_registry_repository" "images" {
  location      = var.region
  repository_id = "rsd-convert"
  format        = "DOCKER"
  depends_on    = [google_project_service.apis]
}

# ── Secret Manager ────────────────────────────────────────────────────────────
resource "google_secret_manager_secret" "openai_key" {
  secret_id = "openai-api-key"
  replication { auto {} }
  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret" "anthropic_key" {
  secret_id = "anthropic-api-key"
  replication { auto {} }
  depends_on = [google_project_service.apis]
}

locals {
  db_url = "postgresql+psycopg://rsd_app:${random_password.db_password.result}@${google_sql_database_instance.postgres.private_ip_address}/rsd"
}

resource "google_secret_manager_secret" "db_url" {
  secret_id = "database-url"
  replication { auto {} }
  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_version" "db_url_v1" {
  secret      = google_secret_manager_secret.db_url.id
  secret_data = local.db_url
}

# ── Service Account for Cloud Run ────────────────────────────────────────────
resource "google_service_account" "run_sa" {
  account_id   = "rsd-run-sa"
  display_name = "rsd-convert Cloud Run service account"
}

resource "google_project_iam_member" "run_secret_access" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.run_sa.email}"
}

resource "google_project_iam_member" "run_sql_client" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.run_sa.email}"
}

# ── Cloud Run service ─────────────────────────────────────────────────────────
resource "google_cloud_run_v2_service" "app" {
  name     = "rsd-convert"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.run_sa.email

    vpc_access {
      connector = google_vpc_access_connector.connector.id
      egress    = "PRIVATE_RANGES_ONLY"
    }

    scaling {
      min_instance_count = 0
      max_instance_count = 10
    }

    timeout = "3600s"

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/rsd-convert/rsd-convert-app:latest"

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
        cpu_idle = true
        startup_cpu_boost = true
      }

      env {
        name  = "STREAMLIT_SERVER_HEADLESS"
        value = "true"
      }
      env {
        name  = "STREAMLIT_BROWSER_GATHER_USAGE_STATS"
        value = "false"
      }

      env {
        name = "OPENAI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.openai_key.secret_id
            version = "latest"
          }
        }
      }

      env {
        name = "ANTHROPIC_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.anthropic_key.secret_id
            version = "latest"
          }
        }
      }

      env {
        name = "DATABASE_URL"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.db_url.secret_id
            version = "latest"
          }
        }
      }

      startup_probe {
        http_get { path = "/_stcore/health" }
        initial_delay_seconds = 10
        period_seconds        = 5
        failure_threshold     = 10
      }

      liveness_probe {
        http_get { path = "/_stcore/health" }
        period_seconds    = 30
        failure_threshold = 3
      }
    }
  }

  depends_on = [
    google_artifact_registry_repository.images,
    google_vpc_access_connector.connector,
  ]
}

# Allow unauthenticated access
resource "google_cloud_run_v2_service_iam_member" "public" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.app.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ── Cloud Build trigger ───────────────────────────────────────────────────────
resource "google_cloudbuild_trigger" "main" {
  name     = "rsd-convert-deploy"
  location = var.region

  github {
    owner = var.github_owner
    name  = var.github_repo
    push {
      branch = "^main$"
    }
  }

  filename = "cloudbuild.yaml"

  substitutions = {
    _REGION  = var.region
    _SERVICE = "rsd-convert"
    _REPO    = "rsd-convert"
    _IMAGE   = "rsd-convert-app"
  }

  depends_on = [google_project_service.apis]
}

# ── Outputs ───────────────────────────────────────────────────────────────────
output "cloud_run_url" {
  description = "URL of the deployed Cloud Run service"
  value       = google_cloud_run_v2_service.app.uri
}

output "artifact_registry_repo" {
  description = "Docker image repo path"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/rsd-convert/rsd-convert-app"
}

output "db_instance_name" {
  description = "Cloud SQL instance name"
  value       = google_sql_database_instance.postgres.name
}

output "openai_secret_name" {
  description = "Secret Manager secret to populate with your OpenAI API key"
  value       = google_secret_manager_secret.openai_key.secret_id
}

output "anthropic_secret_name" {
  description = "Secret Manager secret to populate with your Anthropic API key"
  value       = google_secret_manager_secret.anthropic_key.secret_id
}
