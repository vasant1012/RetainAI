"""Pydantic contracts for the RetainAI API."""
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CustomerFeatures(BaseModel):
    """Raw customer values accepted by the model.

    Known fields are typed for Swagger and validation. Extra fields are allowed so
    a retrained pipeline can add a feature without requiring an immediate API rewrite.
    The model service still rejects fields not present in the trained artifact.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    age: float | None = Field(None, alias="Age", ge=0, le=120)
    gender: str | None = Field(None, alias="Gender", min_length=1, max_length=50)
    country: str | None = Field(None, alias="Country", min_length=1, max_length=100)
    city: str | None = Field(None, alias="City", min_length=1, max_length=100)
    signup_quarter: str | None = Field(
        None, alias="Signup_Quarter", pattern=r"^(Q[1-4]|[1-4])$"
    )
    membership_years: float | None = Field(None, alias="Membership_Years", ge=0)
    login_frequency: float | None = Field(None, alias="Login_Frequency", ge=0)
    session_duration_avg: float | None = Field(
        None, alias="Session_Duration_Avg", ge=0
    )
    pages_per_session: float | None = Field(None, alias="Pages_Per_Session", ge=0)
    cart_abandonment_rate: float | None = Field(
        None, alias="Cart_Abandonment_Rate", ge=0, le=1
    )
    wishlist_items: int | None = Field(None, alias="Wishlist_Items", ge=0)
    total_purchases: int | None = Field(None, alias="Total_Purchases", ge=0)
    average_order_value: float | None = Field(
        None, alias="Average_Order_Value", ge=0
    )
    days_since_last_purchase: float | None = Field(
        None, alias="Days_Since_Last_Purchase", ge=0
    )
    discount_usage_rate: float | None = Field(
        None, alias="Discount_Usage_Rate", ge=0, le=1
    )
    returns_rate: float | None = Field(None, alias="Returns_Rate", ge=0, le=1)
    email_open_rate: float | None = Field(
        None, alias="Email_Open_Rate", ge=0, le=1
    )
    customer_service_calls: int | None = Field(
        None, alias="Customer_Service_Calls", ge=0
    )
    product_reviews_written: int | None = Field(
        None, alias="Product_Reviews_Written", ge=0
    )
    social_media_engagement_score: float | None = Field(
        None, alias="Social_Media_Engagement_Score", ge=0
    )
    mobile_app_usage: float | None = Field(None, alias="Mobile_App_Usage", ge=0)
    payment_method_diversity: int | None = Field(
        None, alias="Payment_Method_Diversity", ge=0
    )
    credit_balance: float | None = Field(None, alias="Credit_Balance")
    lifetime_value: float | None = Field(None, alias="Lifetime_Value", ge=0)

    @model_validator(mode="after")
    def require_a_feature(self) -> "CustomerFeatures":
        if not self.model_dump(exclude_none=True):
            raise ValueError("At least one customer feature must be supplied.")
        return self

    def to_model_payload(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, exclude_none=True)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str | None = Field(None, min_length=1, max_length=128)
    features: CustomerFeatures


class PredictionResponse(BaseModel):
    prediction_id: UUID
    customer_id: str | None
    churn_risk_score: float = Field(ge=0, le=100)
    risk_level: str
    recommended_action: str
    model_version: str


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model_ready: bool


class MetricsResponse(BaseModel):
    model_version: str
    metrics: dict[str, Any]


class ModelInfoResponse(BaseModel):
    ready: bool
    model_version: str | None
    trained_at_utc: str | None
    raw_features: list[str]
